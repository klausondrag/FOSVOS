from __future__ import division

import sys

from pathlib import Path as P
from config.mypath import Path

if Path.is_custom_pytorch():
    sys.path.append(Path.custom_pytorch())  # Custom PyTorch
if Path.is_custom_opencv():
    sys.path.insert(0, Path.custom_opencv())

import numpy as np
import cv2
from scipy.misc import imresize
import os
from torch.utils.data import Dataset


class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, mode='train',
                 inputRes=None,
                 db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.mode = mode.lower()
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        mode_fname_mapping = {
            'train': 'train',
            'val': 'trainval',
            'test': 'val',
        }
        if self.mode in mode_fname_mapping:
            fname = mode_fname_mapping[self.mode]
        else:
            raise Exception('Mode {} does not exist. Must be one of [\'train\', \'val\', \'test\']')

        if self.seq_name is None:
            path_db_root = P(db_root_dir)
            path_sequences = path_db_root / 'ImageSets' / '480p'
            file_extension = '.txt'

            sequences_file = path_sequences / (fname + file_extension)
            with open(str(sequences_file)) as f:
                sequences = f.readlines()
                # sequences[0] == '/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png '
                sequences = [s.split() for s in sequences]
                img_list, labels = zip(*sequences)
                path_db_root.joinpath(*img_list[0].split('/'))

                tmp_list = [i.split('/') for i in img_list]
                seq_list = [i[-2] for i in tmp_list]  # seq_list[0] = bear
                fname_list = [i[-1].split('.')[0] for i in tmp_list]  # seq_list[0] = 00000
                img_list = [str(path_db_root.joinpath(*i.split('/')))
                            for i in img_list]
                labels = [str(P(*l.split('/')))
                          for l in labels]

                # # Initialize the original DAVIS splits for training the parent network
                # with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                #     seqs = f.readlines()
                #     img_list = []
                #     labels = []
                #     for seq in seqs:
                #         images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                #         images_path = map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images)
                #         img_list.extend(images_path)
                #         lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                #         lab_path = map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab)
                #         labels.extend(lab_path)
        else:

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img)
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
            labels.extend([None] * (len(names_img) - 1))
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.seq_list = seq_list
        self.fname_list = fname_list
        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.seq_name is None:
            sample['seq_name'] = self.seq_list[idx]
            sample['fname'] = self.fname_list[idx]
        else:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            # inputRes = list(reversed(self.inputRes))
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt = gt / np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    from dataloaders.custom_transforms import RandomHorizontalFlip, Resize, ToTensor
    from dataloaders.helpers import *

    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    # transforms = transforms.Compose([RandomHorizontalFlip(),
    #                                  ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25))])
    transforms = transforms.Compose([RandomHorizontalFlip(), Resize(scales=[0.5, 0.8, 1]), ToTensor()])

    dataset = DAVIS2016(db_root_dir=Path.db_root_dir(),
                        mode='train', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        if i == 10:
            break

    plt.show(block=True)
