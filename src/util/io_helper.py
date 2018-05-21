import datetime
import socket
from pathlib import Path
from typing import Optional

import shutil
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
from util.logger import get_logger

log = get_logger(__file__)


def visualize_network(net):
    x = torch.randn(1, 3, 480, 854)
    x = Variable(x)
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()


def get_summary_writer(path_tensorboard: Path, delete_dir: bool = True) -> SummaryWriter:
    if delete_dir and path_tensorboard.exists():
        log.warn('Deleting existing tensorboard directory: %s', str(path_tensorboard))
        try:
            shutil.rmtree(str(path_tensorboard))
        except:
            log.warn('Failed to delete the directory')

    path_tensorboard = path_tensorboard / _get_timestamp()
    path_tensorboard = str(path_tensorboard)
    log.info('Logging for tensorboard in directory: %s', path_tensorboard)
    summary_writer = SummaryWriter(path_tensorboard)
    return summary_writer


def _get_timestamp() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def write_settings(save_dir: Path, name: str, settings: Settings, variant_offline: Optional[int] = None,
                   variant_online: Optional[int] = None) -> None:
    if variant_offline is not None:
        name += '_' + str(variant_offline)
        if variant_online is not None:
            name += '_' + str(variant_online)
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
