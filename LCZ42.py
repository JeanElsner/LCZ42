import h5py
from PIL import Image
import os
import os.path
import torch.utils.data as data


class LCZ42(data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.f = h5py.File(os.path.join(os.path.expanduser(root), 'training.h5'), 'r')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        sen1, sen2, label = self.f['sen1'][index], self.f['sen2'][index], self.f['label'][index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        sen1 = self.__extract_images_(sen1)
        sen2 = self.__extract_images_(sen2)
        sen1.extend(sen2)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return sen1, label

    def __extract_images_(self, arr):
        imgs = []
        for i in range(arr.shape[2]):
            img = Image.fromarray(arr[:, :, i])
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        return imgs


    def __len__(self):
        return len(self.f['label'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
