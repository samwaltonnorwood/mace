# This file loads .xyz datasets and prepares HDF5 files that are suitable for training with on-the-fly dataloading

import logging
import ast
import numpy as np
import json
import random
import tqdm
from glob import glob
import h5py
import torch
import multiprocessing
import os
from typing import List, Tuple
from functools import partial

from mace import tools, data
from mace.data.utils import save_configurations_as_HDF5
from mace.tools.scripts_utils import get_dataset_from_xyz, get_atomic_energies
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric
from mace.modules import compute_statistics


def compute_stats(
    file: str, 
    z_table: AtomicNumberTable, 
    r_max: torch.Tensor, 
    atomic_energies: Tuple, 
    batch_size: int
) -> List:
    train_dataset = data.HDF5Dataset(file, z_table=z_table, r_max=r_max)
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False,
    )
    avg_num_neighbors, mean, std = compute_statistics(train_loader, atomic_energies)
    output = [avg_num_neighbors, mean, std]
    return output


def compute_stats_parallel(
    prefix: str, 
    z_table: AtomicNumberTable, 
    r_max: torch.Tensor, 
    atomic_energies: Tuple, 
    batch_size: int,
    num_process: int,
) -> np.ndarray: 
    pool = multiprocessing.Pool(processes=num_process)
    pool_result = [
        pool.apply_async(
            compute_stats, 
            args=(file, z_table, r_max, atomic_energies, batch_size)
        ) 
        for file in glob(prefix+'/*')
    ]
    pool.close()
    pool.join()
    results = [r.get() for r in tqdm.tqdm(pool_result)]
    return np.average(results, axis=0)


def write_hdf5(process, prefix, split_data, drop_last):
    with h5py.File(prefix + str(process) + ".h5", "w") as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(split_data[process], process, f)


def write_hdf5_parallel(data, num_process, prefix, shuffle=False):
    if shuffle:
        data = random.shuffle(data)
    drop_last = (len(data) % 2 == 1)
    split_data = np.array_split(data, num_process)
    processes = []
    for i in range(num_process):
        p = multiprocessing.Process(
            args=[i],
            target=partial(
                write_hdf5,
                prefix=prefix, 
                split_data=split_data, 
                drop_last=drop_last,
            )
        )
        p.start()
        processes.append(p)
    for i in processes:
        i.join()


def split_array(a: np.ndarray, max_size: int):
    drop_last = False
    if len(a) % 2 == 1:
        a = np.append(a, a[-1])
        drop_last = True
    factors = get_prime_factors(len(a))
    max_factor = 1
    for i in range(1, len(factors) + 1):
        for j in range(0, len(factors) - i + 1):
            if np.prod(factors[j : j + i]) <= max_size:
                test = np.prod(factors[j : j + i])
                if test > max_factor:
                    max_factor = test
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


def main():
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    args = tools.build_preprocess_arg_parser().parse_args()
    
    # Setup
    tools.set_seeds(args.seed)
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    r_max = ast.literal_eval(args.r_max)
    if isinstance(r_max, (list, tuple, np.ndarray)):
        r_max = max(r_max)
    r_max = torch.tensor(r_max, dtype=torch.get_default_dtype())
    
    folders = ['train', 'val', 'test']
    for sub_dir in folders:
        if not os.path.exists(args.h5_prefix+sub_dir):
            os.makedirs(args.h5_prefix+sub_dir)
    
    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
    )

    # Atomic number table
    # yapf: disable
    if args.atomic_numbers is None:
        z_table = tools.get_atomic_number_table_from_zs(
            z
            for configs in (collections.train, collections.valid)
            for config in configs
            for z in config.atomic_numbers
        )
    else:
        logging.info("Using atomic numbers from command line argument")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)

    logging.info("Preparing training set")
    write_hdf5_parallel(
        data=collections.train,
        prefix=args.h5_prefix + "train/train_",
        num_process=args.num_process,
        shuffle=args.shuffle,
    )    
    logging.info("Preparing validation set")
    write_hdf5_parallel(
        data=collections.valid, 
        prefix=args.h5_prefix + "val/val_",
        num_process=args.num_process,
        shuffle=args.shuffle,
    )
    if args.test_file is not None:            
        logging.info("Preparing test sets")
        for name, subset in collections.tests:
            write_hdf5_parallel(
                data=subset, 
                prefix=args.h5_prefix + "test/" + name + "_",
                num_process=args.num_process,
            )
    
    if args.compute_statistics:
        logging.info("Computing statistics")
        if len(atomic_energies_dict) == 0:
            atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
        atomic_energies = np.array([atomic_energies_dict[z] for z in z_table.zs])
        (
            avg_num_neighbors, 
            mean, 
            std,
        ) = compute_stats_parallel(
            prefix=args.h5_prefix+'train',
            z_table=z_table,
            r_max=r_max,
            atomic_energies=atomic_energies,
            batch_size=args.batch_size,
            num_process=args.num_process,
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")
        logging.info(f"Average number of neighbors: {avg_num_neighbors}")
        logging.info(f"Mean: {mean}")
        logging.info(f"Standard deviation: {std}")
        # save the statistics as a json
        statistics = {
            "atomic_energies": str(atomic_energies_dict),
            "avg_num_neighbors": avg_num_neighbors,
            "mean": mean,
            "std": std,
            "atomic_numbers": str(z_table.zs),
            "r_max": torch.max(r_max).item(),
        }
        with open(args.h5_prefix + "statistics.json", "w") as f:
            json.dump(statistics, f)


if __name__ == "__main__":
    main()
