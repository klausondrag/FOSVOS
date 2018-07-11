from typing import Optional
import time

import click
import numpy as np
import cv2
import torch

from networks.osvos_resnet import OSVOS_RESNET
from networks.osvos_vgg import OSVOS_VGG
from util.logger import get_logger

log = get_logger(__file__)

mean_value = np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)


@click.command()
@click.option('--variant', '-var', type=click.Choice(['vgg', 'resnet', 'prune', 'mimic']), default='resnet')
@click.option('--version', '-ver', type=int)
@click.option('--webcam', '-wc', type=int, default=0)
@click.option('--mirror/--no-mirror', '-m/-nm', default=True)
@click.option('--use-network/--no-network', '-n/-nn', default=True)
@click.option('--use-cuda/--no-cuda', '-c/-nc', default=True)
@click.option('--overlay/--no-overlay', '-o/-no', default=True)
@click.option('--boolean-mask/--no-boolean-mask', '-bm/-nbm', default=True)
@click.option('--overlay-color', '-oc', type=click.Choice(['r', 'g', 'b']), default='r')
@click.option('--overlay-alpha', '-oa', type=float, default=1.0)
def main(variant: str, version: int, webcam: int, mirror: bool, use_network: bool, use_cuda: bool,
         overlay: bool, boolean_mask: bool, overlay_color: str, overlay_alpha: int) -> None:
    if use_network:
        net = get_network(variant, version)
        if use_cuda:
            net = net.cuda()
    else:
        net = None
    cam = cv2.VideoCapture(webcam)
    loop_video(variant, net, cam, mirror, use_cuda, overlay, boolean_mask, overlay_color, overlay_alpha)
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


def loop_video(variant: str, net: Optional[torch.nn.Module], cam: cv2.VideoCapture, mirror: bool, use_cuda: bool,
               overlay: bool, boolean_mask: bool, overlay_color: str, overlay_alpha: int) -> None:
    use_network = net is not None
    while True:
        start_time = time.time()
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        if use_network:
            img = apply_network(net, img, use_cuda, overlay, boolean_mask, overlay_color, overlay_alpha)
        cv2.imshow(variant, img)
        log.info('FPS: {0:0.1f}'.format(1.0 / (time.time() - start_time)))
        if cv2.waitKey(1) == 27:
            break  # esc to quit


def apply_network(net: torch.nn.Module, img: np.ndarray, use_cuda: bool, overlay: bool,
                  boolean_mask: bool, overlay_color: str, overlay_alpha: int) -> np.ndarray:
    input_img = img
    img = img - mean_value
    img = to_tensor(img)
    if use_cuda:
        img = img.cuda()
    network_output = net.forward(img)
    prediction = network_output[-1]
    prediction = to_numpy(prediction)
    if boolean_mask:
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
    if overlay:
        output = perform_overlay(input_img, prediction, overlay_alpha, overlay_color)
    else:
        output = prediction
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


def perform_overlay(input_img: np.ndarray, prediction: np.ndarray, overlay_alpha: float,
                    overlay_color: str) -> np.ndarray:
    if overlay_color == 'r':
        color_index = 2
    elif overlay_color == 'g':
        color_index = 1
    elif overlay_color == 'b':
        color_index = 0
    else:
        raise Exception('Click should have prevented this')
    mask = np.zeros(input_img.shape, dtype=float)
    mask[..., color_index] = 255
    output = input_img + overlay_alpha * mask * prediction[..., np.newaxis]
    output[output > 255] = 255
    output = output.astype('uint8')
    return output


if __name__ == '__main__':
    main()
