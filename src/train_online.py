import socket
import sys
import timeit
from datetime import datetime
from pathlib import Path

from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloaders.davis_2016 import DAVIS2016
from dataloaders import custom_transforms
import visualize as viz
import scipy.misc as sm
import networks.osvos_vgg as vo
from layers.osvos_layers import class_balanced_cross_entropy_loss
from dataloaders.helpers import *
from util import gpu_handler
from util.logger import get_logger
from config.mypath import Path as P
from util.network_provider import NetworkProvider

if P.is_custom_pytorch():
    sys.path.append(P.custom_pytorch())  # Custom PyTorch
gpu_handler.select_gpu_by_hostname()

log = get_logger(__file__)

db_root_dir = P.db_root_dir()
exp_dir = P.exp_dir()
is_visualizing_network = False
is_visualizing_result = False
n_avg_grad = 5
start_epoch = 0
snapshot_every = 100  # Store a model every snapshot epochs
parent_epoch = 240  # 240

# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
}

save_dir = Path('models')
save_dir.mkdir(exist_ok=True)
net_provider = NetworkProvider('vgg16_blackswan', vo.OSVOS_VGG, save_dir)

if is_visualizing_result:
    import matplotlib.pyplot as plt


def get_optimizer(net, learning_rate: float = 1e-8, weight_decay: float = 0.0002):
    optimizer = optim.SGD([
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': weight_decay},
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': learning_rate * 2},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]],
         'weight_decay': weight_decay},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': learning_rate * 2},
        {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': net.fuse.weight, 'lr': learning_rate / 100, 'weight_decay': weight_decay},
        {'params': net.fuse.bias, 'lr': 2 * learning_rate / 100},
    ], lr=learning_rate, momentum=0.9)
    return optimizer


def train(seq_name: str, n_epochs: int, name_parent: str = 'vgg16', train_and_test: bool = True) -> None:
    speeds_training = []
    if train_and_test:
        # Network definition
        net_provider.name = name_parent + '_' + seq_name
        net = net_provider.init_network(pretrained=0)
        net_provider.load(parent_epoch, name=name_parent)

        # Logging into Tensorboard
        log_dir = save_dir / 'runs' / (datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()
                                       + '-' + seq_name)
        writer = SummaryWriter(log_dir=str(log_dir))

        # Visualize the network
        if is_visualizing_network:
            x = torch.randn(1, 3, 480, 854)
            x = Variable(x)
            x = gpu_handler.cast_cuda_if_possible(x)
            y = net.forward(x)
            g = viz.make_dot(y, net.state_dict())
            g.view()

        optimizer = get_optimizer(net)

        # Preparation of the data loaders
        # Define augmentation transformations as a composition
        composed_transforms = transforms.Compose([custom_transforms.RandomHorizontalFlip(),
                                                  custom_transforms.Resize(),
                                                  # custom_transforms.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                                  custom_transforms.ToTensor()])
        # Training dataset and its iterator
        db_train = DAVIS2016(mode='train', db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
        trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)

        # Testing dataset and its iterator
        db_test = DAVIS2016(mode='test', db_root_dir=db_root_dir, transform=custom_transforms.ToTensor(),
                            seq_name=seq_name)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

        num_img_tr = len(trainloader)
        num_img_ts = len(testloader)
        loss_tr = []
        counter_gradient = 0

        log.info("Start of Online Training, sequence: " + seq_name)
        start_time = timeit.default_timer()
        # Main Training and Testing Loop
        for epoch in range(start_epoch, n_epochs):
            epoch_start_time = timeit.default_timer()
            # One training epoch
            running_loss_tr = 0
            for ii, sample_batched in enumerate(trainloader):

                inputs, gts = sample_batched['image'], sample_batched['gt']

                # Forward-Backward of the mini-batch
                inputs, gts = Variable(inputs), Variable(gts)
                inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

                outputs = net.forward(inputs)

                # Compute the fuse loss
                loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
                running_loss_tr += loss.data[0]

                # Print stuff
                if epoch % (n_epochs // 20) == (n_epochs // 20 - 1):
                    running_loss_tr /= num_img_tr
                    loss_tr.append(running_loss_tr)

                    log.info('[Epoch {0}: {1}, numImages: {2}]'.format(seq_name, epoch + 1, ii + 1))
                    log.info('Loss {0}: {1}'.format(seq_name, running_loss_tr))
                    writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

                # Backward the averaged gradient
                loss /= n_avg_grad
                loss.backward()
                counter_gradient += 1

                # Update the weights once in nAveGrad forward passes
                if counter_gradient % n_avg_grad == 0:
                    writer.add_scalar('data/total_loss_iter', loss.data[0], ii + num_img_tr * epoch)
                    optimizer.step()
                    optimizer.zero_grad()
                    counter_gradient = 0

            # Save the model
            if (epoch % snapshot_every) == snapshot_every - 1:  # and epoch != 0:
                net_provider.save(epoch)

            epoch_stop_time = timeit.default_timer()
            t = epoch_stop_time - epoch_start_time
            log.info('epoch {0} {1}: {2} sec'.format(seq_name, str(epoch), str(t)))
            speeds_training.append(t)

        stop_time = timeit.default_timer()
        log.info('Train {0}: total training time {1} sec'.format(seq_name, str(stop_time - start_time)))
        log.info('Train {0}: time per sample {1} sec'.format(seq_name, np.asarray(t).mean()))

        # Testing Phase
        if is_visualizing_result:
            ax_arr = init_plot()
    else:
        net_provider.name = name_parent + '_' + seq_name
        net = net_provider.init_network(pretrained=0)
        net_provider.load(parent_epoch)

        db_test = DAVIS2016(mode='test', db_root_dir=db_root_dir, transform=custom_transforms.ToTensor(),
                            seq_name=seq_name)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    save_dir_res = Path('results') / seq_name
    save_dir_res.mkdir(exist_ok=True)

    log.info('Testing Network')
    test_start_time = timeit.default_timer()
    # Main Testing Loop
    for ii, sample_batched in enumerate(testloader):

        img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

        # Forward of the mini-batch
        inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
        inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

        outputs = net.forward(inputs)

        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)

            # Save the result, attention to the index jj
            log.info(str(fname))
            file_name = save_dir_res / '{0}.png'.format(fname[jj])
            sm.imsave(file_name, pred)

            if is_visualizing_result:
                visualize_results(ax_arr, gt, img, jj, pred)

    test_stop_time = timeit.default_timer()
    log.info('Test {0}: total training time {1} sec'.format(seq_name, str(test_stop_time - test_start_time)))
    log.info('Test {0}: {1} images'.format(seq_name, str((len(testloader)))))
    log.info(
        'Test {0}: time per sample {1} sec'.format(seq_name, str((test_stop_time - test_start_time) / len(testloader))))


def init_plot():
    plt.close("all")
    plt.ion()
    f, ax_arr = plt.subplots(1, 3)
    return ax_arr


def visualize_results(ax_arr, gt, img, jj, pred):
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
    n_epochs = 400 * n_avg_grad  # Number of epochs for training

    sequences = ['bear', 'boat', 'camel', 'cows', 'dog-agility', 'elephant', 'hockey', 'kite-walk', 'mallard-water',
                 'paragliding', 'rollerblade', 'soccerball', 'tennis', 'blackswan', 'breakdance', 'car-roundabout',
                 'dance-jump', 'drift-chicane', 'flamingo', 'horsejump-high', 'libby', 'motocross-bumps',
                 'paragliding-launch', 'scooter-black', 'stroller', 'train', 'bmx-bumps', 'breakdance-flare',
                 'car-shadow', 'dance-twirl', 'drift-straight', 'goat', 'horsejump-low', 'lucia', 'motocross-jump',
                 'parkour', 'scooter-gray', 'surf', 'bmx-trees', 'bus', 'car-turn', 'dog', 'drift-turn', 'hike',
                 'kite-surf', 'mallard-fly', 'motorbike', 'rhino', 'soapbox', 'swing']
    already_done = []
    # already_done = ['bear', 'blackswan', 'boat', 'camel', 'cows', 'dog-agility', 'elephant', 'hockey']
    sequences = [s for s in sequences if s not in already_done]

    [train(s, n_epochs) for s in sequences]
    # train('boat', n_epochs, train_and_test=False)
