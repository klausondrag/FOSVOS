# https://gist.github.com/tedmiston/6060034
"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import click
import numpy as np
import cv2
import torch
from torch.autograd import Variable

from networks.osvos_resnet import OSVOS_RESNET
from networks.osvos_vgg import OSVOS_VGG


@click.command()
@click.option('--variant', type=click.Choice(['vgg16', 'resnet34', 'resnet18', 'prune60', 'mimic3']),
              default='prune60')
@click.option('--webcam', type=int, default=0)
@click.option('--mirror/--no-mirror', default=True)
def main(variant: str, webcam: int, mirror: bool) -> None:
    net = get_network(variant)
    cam = cv2.VideoCapture(webcam)
    loop_video(net, cam, mirror)
    cv2.destroyAllWindows()


def get_network(variant: str) -> torch.nn.Module:
    if variant == 'vgg16':
        net = OSVOS_VGG(pretrained=False)
        net.load_state_dict(torch.load('vgg16.pth', map_location=lambda storage, loc: storage))
    elif variant == 'resnet34':
        raise Exception('Not yet implemented')
    elif variant == 'resnet18':
        net = OSVOS_RESNET(pretrained=False)
        net.load_state_dict(torch.load('resnet18.pth', map_location=lambda storage, loc: storage))
    elif variant == 'prune60':
        net = torch.load('prune_64_1_60.pth', map_location=lambda storage, loc: storage)
    elif variant == 'mimic3':
        raise Exception('Not yet implemented')
    else:
        raise Exception('Click should have prevented this')
    net = net.cuda()
    return net


def loop_video(net: torch.nn.Module, cam: cv2.VideoCapture, mirror: bool) -> None:
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        img = apply_network(net, img, use_net=True)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit


def apply_network(net: torch.nn.Module, img: np.ndarray, use_net: bool) -> np.ndarray:
    img = img[np.newaxis, ...]
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    if isinstance(img, torch.ByteTensor):
        img = img.float().div(255)
    img = Variable(img, volatile=True)
    img = img.cuda()
    if use_net:
        outputs = net.forward(img)
        pred = outputs[-1]
    else:
        pred = img
    pred = pred.cpu().data.numpy()[0, :, :, :]
    pred = np.transpose(pred, (1, 2, 0))
    pred = 1 / (1 + np.exp(-pred))
    pred = np.squeeze(pred)
    return pred


if __name__ == '__main__':
    main()
