import datetime
import socket
from pathlib import Path
from typing import Optional, Dict

import torch
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import visualize as viz
from dataloaders import custom_transforms
from dataloaders.davis_2016 import DAVIS2016
from util.settings import Settings


def visualize_network(net):
    x = torch.randn(1, 3, 480, 854)
    x = Variable(x)
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()


def get_summary_writer(save_dir: Path, postfix: Optional[str] = None, comment: str = '') -> SummaryWriter:
    name_parts = [_get_timestamp(), socket.gethostname()]
    if postfix is not None:
        name_parts.append(postfix)
    dir_name = '_'.join(name_parts)
    log_dir = save_dir / 'runs' / dir_name
    summary_writer = SummaryWriter(log_dir=str(log_dir), comment=comment)
    return summary_writer


def _get_timestamp() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def write_settings(save_dir: Path, name: str, settings: Settings) -> None:
    file_name = '{0}_settings_{1}.yml'.format(name, _get_timestamp())
    file_path = save_dir / file_name
    with open(str(file_path), 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)


def get_data_loader_train(db_root_dir: Path, batch_size: int, seq_name: Optional[str] = None) -> DataLoader:
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([custom_transforms.RandomHorizontalFlip(),
                                              custom_transforms.Resize(),
                                              # custom_transforms.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              custom_transforms.ToTensor()])
    db_train = DAVIS2016(mode='train', db_root_dir=str(db_root_dir), transform=composed_transforms, seq_name=seq_name)
    data_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1)
    return data_loader


def get_data_loader_test(db_root_dir: Path, batch_size: int, seq_name: Optional[str] = None) -> DataLoader:
    db_test = DAVIS2016(mode='test', db_root_dir=str(db_root_dir), transform=custom_transforms.ToTensor(),
                        seq_name=seq_name)
    data_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return data_loader
