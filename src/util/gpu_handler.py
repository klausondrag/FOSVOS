import socket
from typing import Optional, Union, List

import torch

from util.logger import get_logger

log = get_logger(__file__)

_hostname_to_gpu_id = {
    'eec': 1,
    'hpccremers6': 1
}

_gpu_id_default_value = -1


def select_gpu_by_id(gpu_id: int = _gpu_id_default_value) -> None:
    torch.cuda.set_device(device=gpu_id)


def select_gpu_by_hostname(hostname: Optional[str] = None) -> None:
    if hostname is None:
        hostname = socket.gethostname()
    gpu_id = _hostname_to_gpu_id.get(hostname, _gpu_id_default_value)
    select_gpu_by_id(gpu_id)


def cast_cuda_if_possible(net: Union[List[torch.nn.Module], torch.nn.Module],
                          verbose: bool = False) -> Union[List[torch.nn.Module], torch.nn.Module]:
    if torch.cuda.is_available():
        if verbose:
            log.info('Using cuda')
        if type(net) is list:
            return [n.cuda() for n in net]
        else:
            return net.cuda()
    else:
        if verbose:
            log.warn('Not using cuda')
        return net


if __name__ == '__main__':
    select_gpu_by_id(0)
    select_gpu_by_hostname('_')
    select_gpu_by_hostname('hpccremers6')
    select_gpu_by_hostname()
    exit(0)
