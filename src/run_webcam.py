from typing import Optional
import time

import click
import numpy as np
import cv2
import torch

from networks.osvos_resnet import OSVOS_RESNET
from networks.osvos_vgg import OSVOS_VGG

mean_value = np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)


@click.command()
@click.option('--variant', '-var', type=click.Choice(['vgg', 'resnet', 'prune', 'mimic']), default='resnet')
@click.option('--version', '-ver', type=int)
@click.option('--webcam', '-wc', type=int, default=0)
@click.option('--mirror/--no-mirror', '-m/-nm', default=True)
@click.option('--use-network/--no-network', '-n/-nn', default=True)
@click.option('--use-cuda/--no-cuda', '-c/-nc', default=True)
def main(variant: str, version: int, webcam: int, mirror: bool, use_network: bool, use_cuda: bool) -> None:
    if use_network:
        net = get_network(variant, version)
        if use_cuda:
            net = net.cuda()
    else:
        net = None
    cam = cv2.VideoCapture(webcam)
    loop_video(net, cam, mirror, use_cuda)
    cv2.destroyAllWindows()


def get_network(variant: str, version: int) -> torch.nn.Module:
    if variant == 'vgg':
        net = OSVOS_VGG(pretrained=False)
        net.load_state_dict(torch.load('vgg16.pth', map_location=lambda storage, loc: storage))
    elif variant == 'resnet':
        if version != 34:
            version = 18
        net = OSVOS_RESNET(pretrained=False, version=version)
        net.load_state_dict(torch.load('resnet{}.pth'.format(str(version)), map_location=lambda storage, loc: storage))
    elif variant == 'prune':
        net = torch.load('prune_64_1_{}.pth'.format(version), map_location=lambda storage, loc: storage)
    elif variant == 'mimic':
        raise Exception('Not yet implemented')
    else:
        raise Exception('Click should have prevented this')
    return net


def loop_video(net: Optional[torch.nn.Module], cam: cv2.VideoCapture, mirror: bool, use_cuda: bool) -> None:
    use_network = net is not None
    while True:
        start_time = time.time()
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        if use_network:
            img = apply_network(net, img, use_cuda)
        cv2.imshow('my webcam', img)
        print('FPS: {0:0.1f}'.format(1.0 / (time.time() - start_time)))
        if cv2.waitKey(1) == 27:
            break  # esc to quit


def apply_network(net: torch.nn.Module, img: np.ndarray, use_cuda: bool) -> np.ndarray:
    input_img = img
    img = img - mean_value
    img = to_tensor(img)
    if use_cuda:
        img = img.cuda()
    predictions = net.forward(img)
    prediction = predictions[-1]
    prediction = to_numpy(prediction)

    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    alpha = 1
    mask = np.zeros(input_img.shape, dtype=float)
    mask[..., 2] = 255
    output = input_img + alpha * mask * prediction[..., np.newaxis]
    output[output > 255] = 255
    output = output.astype('uint8')

    return output


def to_tensor(img: np.ndarray) -> torch.autograd.Variable:
    img = img[np.newaxis, ...]
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    if isinstance(img, torch.ByteTensor):
        img = img.float().div(255)
    img = torch.autograd.Variable(img, volatile=True)
    return img


def to_numpy(output: torch.nn.Module) -> np.ndarray:
    output = output.cpu().data.numpy()[0, :, :, :]
    output = np.transpose(output, (1, 2, 0))
    output = 1 / (1 + np.exp(-output))
    output = np.squeeze(output)
    return output


if __name__ == '__main__':
    main()
