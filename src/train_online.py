import sys
import timeit
from pathlib import Path

from tensorboardX import SummaryWriter
import scipy.misc as sm

from torch.autograd import Variable
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from layers.osvos_layers import class_balanced_cross_entropy_loss
from dataloaders.helpers import *
from util import gpu_handler, io_helper
from util.logger import get_logger
from config.mypath import Path as P
from util.network_provider import NetworkProvider, OnlineSettings, VGGOnlineProvider, ResNetOnlineProvider

if P.is_custom_pytorch():
    sys.path.append(P.custom_pytorch())  # Custom PyTorch

gpu_handler.select_gpu_by_hostname()
log = get_logger(__file__)


def train_and_test(net_provider: NetworkProvider, seq_name: str, settings: OnlineSettings,
                   is_training: bool = True, is_testing: bool = True) -> None:
    if is_training:
        net_provider.load_network_train()
        data_loader = io_helper.get_data_loader_train(db_root_dir, settings.batch_size_train, seq_name)
        optimizer = net_provider.get_optimizer()
        summary_writer = _get_summary_writer(seq_name)

        io_helper.write_settings(save_dir_models, net_provider.name, settings)
        _train(net_provider, data_loader, optimizer, summary_writer, seq_name, settings.start_epoch, settings.n_epochs,
               settings.avg_grad_every_n, settings.snapshot_every_n)

    if is_testing:
        net_provider.load_network_test()
        data_loader = io_helper.get_data_loader_test(db_root_dir, settings.batch_size_test, seq_name)
        save_dir = save_dir_results / settings.offline_name / 'online'

        _test(net_provider, data_loader, seq_name, save_dir, settings.is_visualizing_results)

    if settings.is_visualizing_network:
        io_helper.visualize_network(net_provider.network)


def _get_summary_writer(seq_name: str) -> SummaryWriter:
    return io_helper.get_summary_writer(save_dir_models, postfix=seq_name)


def _train(net_provider: NetworkProvider, data_loader: DataLoader, optimizer: Optimizer, summary_writer: SummaryWriter,
           seq_name: str, start_epoch: int, n_epochs: int, avg_grad_every_n: int, snapshot_every_n: int) -> None:
    log.info('Start of Online Training, sequence: ' + seq_name)

    net = net_provider.network

    speeds_training = []
    n_samples = len(data_loader)
    loss_tr = []
    counter_gradient = 0

    start_time = timeit.default_timer()
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = timeit.default_timer()

        running_loss_tr = 0
        for minibatch_index, minibatch in enumerate(data_loader):
            inputs, gts = minibatch['image'], minibatch['gt']
            inputs, gts = Variable(inputs), Variable(gts)
            inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

            outputs = net.forward(inputs)

            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            running_loss_tr += loss.data[0]

            if epoch % (n_epochs // 20) == (n_epochs // 20 - 1):
                running_loss_tr /= n_samples
                loss_tr.append(running_loss_tr)

                log.info('[Epoch {0}: {1}, numImages: {2}]'.format(seq_name, epoch + 1, minibatch_index + 1))
                log.info('Loss {0}: {1}'.format(seq_name, running_loss_tr))
                summary_writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

            loss /= avg_grad_every_n
            loss.backward()
            counter_gradient += 1

            if counter_gradient % avg_grad_every_n == 0:
                summary_writer.add_scalar('data/total_loss_iter', loss.data[0], minibatch_index + n_samples * epoch)
                optimizer.step()
                optimizer.zero_grad()
                counter_gradient = 0

        if (epoch % snapshot_every_n) == snapshot_every_n - 1:  # and epoch != 0:
            net_provider.save_model(epoch, sequence=seq_name)

        epoch_stop_time = timeit.default_timer()
        t = epoch_stop_time - epoch_start_time
        log.info('epoch {0} {1}: {2} sec'.format(seq_name, str(epoch), str(t)))
        speeds_training.append(t)

    stop_time = timeit.default_timer()
    log.info('Train {0}: total training time {1} sec'.format(seq_name, str(stop_time - start_time)))
    log.info('Train {0}: time per sample {1} sec'.format(seq_name, np.asarray(t).mean()))


def _test(net_provider: NetworkProvider, data_loader: DataLoader, seq_name: str, save_dir: Path,
          is_visualizing_results: bool) -> None:
    log.info('Testing Network')

    net = net_provider.network

    if is_visualizing_results:
        ax_arr = _init_plot()

    test_start_time = timeit.default_timer()
    for minibatch in data_loader:

        img, gt, fname = minibatch['image'], minibatch['gt'], minibatch['fname']

        inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
        inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

        outputs = net.forward(inputs)

        for index in range(inputs.size()[0]):
            pred = np.transpose(outputs[-1].cpu().data.numpy()[index, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)

            save_dir_seq = save_dir / seq_name  # don't using index!
            save_dir_seq.mkdir(parents=True, exist_ok=True)

            file_name = save_dir_seq / '{0}.png'.format(fname[index])
            sm.imsave(str(file_name), pred)

            if is_visualizing_results:
                _visualize_results(ax_arr, gt, img, index, pred)

    test_stop_time = timeit.default_timer()
    log.info('Test {0}: total training time {1} sec'.format(seq_name, str(test_stop_time - test_start_time)))
    log.info('Test {0}: {1} images'.format(seq_name, str((len(data_loader)))))
    log.info(
        'Test {0}: time per sample {1} sec'.format(seq_name,
                                                   str((test_stop_time - test_start_time) / len(data_loader))))


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


if __name__ == '__main__':
    db_root_dir = P.db_root_dir()
    exp_dir = P.exp_dir()

    save_dir_models = Path('models')
    save_dir_models.mkdir(parents=True, exist_ok=True)
    save_dir_results = Path('results')
    save_dir_results.mkdir(parents=True, exist_ok=True)

    is_training = True
    is_training = False

    use_vgg = True
    use_vgg = False

    settings = OnlineSettings(is_training=is_training, start_epoch=0, n_epochs=2000, avg_grad_every_n=5,
                              snapshot_every_n=2000, is_testing_while_training=False, test_every_n=5,
                              batch_size_train=1, batch_size_test=1, is_visualizing_network=False,
                              is_visualizing_results=False, offline_epoch=240)

    net_provider = VGGOnlineProvider('vgg16', save_dir_models, settings)
    net_provider = ResNetOnlineProvider('resnet18', save_dir_models, settings)

    if settings.is_visualizing_results:
        import matplotlib.pyplot as plt

    sequences = ['bear', 'blackswan', 'bmx-bumps', 'bmx-trees', 'boat', 'breakdance', 'breakdance-flare', 'bus',
                 'camel', 'car-roundabout', 'car-shadow', 'car-turn', 'cows', 'dance-jump', 'dance-twirl', 'dog',
                 'dog-agility', 'drift-chicane', 'drift-straight', 'drift-turn', 'elephant', 'flamingo', 'goat', 'hike',
                 'hockey', 'horsejump-high', 'horsejump-low', 'kite-surf', 'kite-walk', 'libby', 'lucia', 'mallard-fly',
                 'mallard-water', 'motocross-bumps', 'motocross-jump', 'motorbike', 'paragliding', 'paragliding-launch',
                 'parkour', 'rhino', 'rollerblade', 'scooter-black', 'scooter-gray', 'soapbox', 'soccerball',
                 'stroller', 'surf', 'swing', 'tennis', 'train']

    already_done = []
    sequences = [s for s in sequences if s not in already_done]

    # [train_and_test(net_provider, s, settings, is_training=is_training) for s in sequences]
    train_and_test(net_provider, 'bear', settings, is_training=is_training)
