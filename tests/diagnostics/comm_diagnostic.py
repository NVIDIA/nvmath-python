# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

try:
    from cuda.core import Device, system
except ImportError:
    from cuda.core.experimental import Device, system
import os
import socket
import warnings
from dataclasses import dataclass

import nccl.core as nccl
import numpy as np
from mpi4py import MPI


def initialize_nccl(comm, rank, nranks):
    # Create NCCL communicator.
    unique_id = nccl.get_unique_id()
    comm.Bcast(unique_id.as_ndarray.view(np.int8), root=0)
    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)
    return nccl_comm


def set_device(rank):
    device_id = rank % num_devices
    device = Device(device_id)
    device.set_current()


@dataclass
class HostInfo:
    # Number of devices on this host.
    num_devices: int = 0
    # Number of processes on this host.
    num_procs: int = 0


parser = argparse.ArgumentParser(description="MPI and NCCL diagnostic tool")
parser.add_argument("--nccl", action="store_true", help="Diagnose NCCL")
args = parser.parse_args()
use_nccl = args.nccl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
try:
    num_devices = system.get_num_devices()
except AttributeError:
    num_devices = system.num_devices

if nranks < 2:
    raise RuntimeError(
        "You need to run with multiple processes to take advantage of MGMN. Try running "
        f"this script with `mpiexec -n $num_procs python {os.path.basename(__file__)}`"
    )

cluster_info = comm.allgather((socket.gethostname(), num_devices))

# This check is probably unnecessary because if MPI isn't working, the allgather above
# should fail.
if cluster_info is None or len(cluster_info) != nranks:
    raise RuntimeError("MPI is not working (did not get information from every process)")

# Construct a map of number of processes and devices per host.
host_info = {}
for hostname, host_num_devices in cluster_info:
    if hostname not in host_info:
        host_info[hostname] = HostInfo(num_devices=host_num_devices, num_procs=1)
    else:
        if host_info[hostname].num_devices != host_num_devices:
            raise RuntimeError(f"Processes on host {hostname} are not reporting the same device count")
        host_info[hostname].num_procs += 1

suboptimal_hosts = []
if rank == 0:
    print("\n========== Host info ==========")
for hostname, info in host_info.items():
    if rank == 0:
        print(f"* Host {hostname}: {info}")
    if info.num_devices < info.num_procs:
        suboptimal_hosts.append(hostname)
if rank == 0:
    print("")

comm.Barrier()

if suboptimal_hosts:
    if use_nccl:
        raise RuntimeError(
            "NCCL doesn't allow multiple processes per GPU: run the same number of processes "
            "on each host as number of local GPUs."
        )
    elif rank == 0:
        warnings.warn(
            f"The setup is suboptimal in the following hosts: {suboptimal_hosts}. An optimal "
            "setup requires a CUDA device uniquely assigned to a single process."
        )

if use_nccl:
    set_device(rank)
    nccl_comm = initialize_nccl(comm, rank, nranks)
    if rank == 0:
        print("NCCL initialized")
    nccl_comm.destroy()
    if rank == 0:
        print("NCCL communicator destroyed")

if rank == 0:
    print("\nMPI test passed")
    if use_nccl:
        print("NCCL test passed")
