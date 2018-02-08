import timeit
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import cuda

from dataloaders.helpers import im_normalize
from . import gpu_handler
from .network_provider import NetworkProvider
from .logger import get_logger

log = get_logger(__file__)


def test(net_provider: NetworkProvider, data_loader: DataLoader, save_dir: Path,
         is_visualizing_results: bool, eval_speeds: bool, seq_name: Optional[str] = None) -> None:
    log.info('Testing Network')

    net = net_provider.network

    if is_visualizing_results:
        ax_arr = _init_plot()

    n_runs = 1
    times = []
    if eval_speeds:
        n_runs = 10
    time_all_start = timeit.default_timer()
    for _ in range(n_runs):
        for minibatch_index, minibatch in enumerate(data_loader):
            img, gt, minibatch_seq_name, fname = minibatch['image'], minibatch['gt'], \
                                                 minibatch['seq_name'], minibatch['fname']

            inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
            inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

            if eval_speeds:
                # https://github.com/jcjohnson/cnn-benchmarks/blob/master/utils.lua
                cuda.synchronize()
                time_image_start = timeit.default_timer()
            outputs = net.forward(inputs)
            if eval_speeds:
                cuda.synchronize()
                time_image_stop = timeit.default_timer()
                time_image_total = time_image_stop - time_image_start
                if minibatch_index > 0:
                    # first allocate takes longer
                    times.append(time_image_total)
            else:
                for index in range(inputs.size()[0]):
                    pred = np.transpose(outputs[-1].cpu().data.numpy()[index, :, :, :], (1, 2, 0))
                    pred = 1 / (1 + np.exp(-pred))
                    pred = np.squeeze(pred)

                    save_dir_seq = save_dir / minibatch_seq_name[index]
                    save_dir_seq.mkdir(parents=True, exist_ok=True)

                    file_name = save_dir_seq / '{0}.png'.format(fname[index])
                    misc.imsave(str(file_name), pred)

                    if is_visualizing_results:
                        _visualize_results(ax_arr, gt, img, index, pred)

    time_all_stop = timeit.default_timer()
    time_for_all = time_all_stop - time_all_start
    n_images = len(data_loader)
    time_per_sample = time_for_all / n_images
    log.info('Test {0}: total test time {1} sec'.format(seq_name, str(time_for_all)))
    log.info('Test {0}: {1} images'.format(seq_name, str(n_images)))
    log.info('Test {0}: time per sample {1} sec'.format(seq_name, str(time_per_sample)))

    if eval_speeds:
        log.info('Test {0}: accurate {1} images'.format(seq_name, str((n_images - 1) * n_runs)))
        log.info('Test {0}: accurate total time {1} sec ({2} runs)'.format(seq_name, np.sum (times), n_runs))
        log.info('Test {0}: accurate time per sample {1} sec ({2} runs)'.format(seq_name, np.average(times), n_runs))


def _init_plot():
    plt.close('all')
    plt.ion()
    f, ax_arr = plt.subplots(1, 3)
    return ax_arr


def _visualize_results(ax_arr, gt, img, jj, pred):
    img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
    gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
    gt_ = np.squeeze(gt)
    # Plot the particular example
    ax_arr[0].cla()
    ax_arr[1].cla()
    ax_arr[2].cla()
    ax_arr[0].set_title('Input Image')
    ax_arr[1].set_title('Ground Truth')
    ax_arr[2].set_title('Detection')
    ax_arr[0].imshow(im_normalize(img_))
    ax_arr[1].imshow(gt_)
    ax_arr[2].imshow(im_normalize(pred))
    plt.pause(0.001)
