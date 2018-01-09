import datetime
import socket
from pathlib import Path
from typing import Optional, Dict

import torch
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import visualize as viz


def visualize_network(net):
    x = torch.randn(1, 3, 480, 854)
    x = Variable(x)
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()


def get_summary_writer(save_dir: Path, postfix: Optional[str] = None, comment: str = '') -> SummaryWriter:
    name_parts = [_get_timestamp(), socket.gethostname(), postfix]
    dir_name = '_'.join(name_parts)  # if postfix is None it will be skipped
    log_dir = save_dir / 'runs' / dir_name
    summary_writer = SummaryWriter(log_dir=str(log_dir), comment=comment)
    return summary_writer


def _get_timestamp() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def write_settings(save_dir: Path, name: str, settings: Dict) -> None:
    file_name = '{0}_{1}_settings.yml'.format(name, _get_timestamp())
    file_path = save_dir / file_name
    with open(str(file_path), 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)
