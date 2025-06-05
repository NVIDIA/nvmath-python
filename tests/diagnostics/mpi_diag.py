import socket
import cuda.core.experimental
import os
import warnings

from dataclasses import dataclass
from mpi4py import MPI


@dataclass
class HostInfo:
    num_devices: int = 0
    num_procs: int = 0


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
num_devices = cuda.core.experimental.system.num_devices

if nranks < 2:
    raise RuntimeError(
        "You need to run with multiple processes to take advantage of MGMN. Try running "
        f"this script with `mpiexec -n $num_procs python {os.path.basename(__file__)}`"
    )

cluster_info = comm.gather((socket.gethostname(), num_devices))

# NOTE: do no more communication after this point due to rank 0 being the only
# one raising exceptions.

if rank == 0:
    if len(cluster_info) != nranks:
        # Note: if MPI gather failed, it is more likely that the comm.gather
        # raised an exception.
        raise RuntimeError("MPI is not working (did not get information from every process)")

    # Construct a map of number of processes and devices per host.
    host_info = {}
    for hostname, num_devices in cluster_info:
        if hostname not in host_info:
            host_info[hostname] = HostInfo(num_devices=num_devices, num_procs=1)
        else:
            if host_info[hostname].num_devices != num_devices:
                raise RuntimeError(f"Processes on host {hostname} are not reporting the same device count")
            host_info[hostname].num_procs += 1

    suboptimal_hosts = []
    print("\n========== Host info ==========")
    for hostname, info in host_info.items():
        print(f"- Host {hostname}: num_procs={info.num_procs} num_devices={info.num_devices}")
        if info.num_devices < info.num_procs:
            suboptimal_hosts.append(hostname)
    print("")

    if len(suboptimal_hosts) > 0:
        warnings.warn(
            f"The setup is suboptimal in the following hosts: {suboptimal_hosts}. An optimal "
            "setup requires a CUDA device uniquely assigned to a single process."
        )
    else:
        print("No issues found with the current MPI setup for running MGMN operations with nvmath.distributed")

    print("\nMPI test passed")
