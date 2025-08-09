import logging
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import h5py
from util import PalGenWrapper
from torchvision import transforms



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),  # Converts to tensor and normalizes to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Optional: normalize to [-1, 1]
])



class ArCDataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.hdf5 = None # we will lazily init this

        if not transform:
            self.transform = transform

    def __len__(self):
        if not self.hdf5:
            self.hdf5 = h5py.File(self.hdf5_path, "r")
            self.images = self.hdf5['input']
            self.targets = self.hdf5['targets']
        return len(self.images)


    def __getitem__(self, idx):
        if not self.hdf5:
            self.hdf5 = h5py.File(self.hdf5_path, "r")
            self.images = self.hdf5['input']
            self.targets = self.hdf5['target']

        num_colors = random.randrange(1,64,1)
        
        palette = self.targets[idx]
        img = self.images[idx]

        palette = palette[:num_colors]
        img = self.transform(img) if self.transform else img

        target_trch = torch.from_numpy(palette)

        return (img, target_trch)


        



def one_hot_encoding(rgb_np, num_classes):
    N, C = rgb_np.shape
    one_hot_mask = np.zeros((num_classes, *rgb_np.shape), dtype=np.uint8)

    for cls in range(num_classes):
        one_hot_mask[cls][rgb_np == cls] = 1

    one_hot_mask = np.moveaxis(one_hot_mask, [0, 1, 2], [2, 0, 1])
    return one_hot_mask


def add_data_hdf5(image_path, hdf5, logger, palgen, idx):

    img = Image.open(image_path)
    img = img.convert('RGB')
    logger.info(f"resizing {image_path}")
    new_size = (256,256)
    resized_img = img.resize(new_size)
    img_np = np.array(resized_img)
    img_np = np.swapaxes(img_np, 0, 2)
    logger.info(f"Getting palette of {image_path} from Palgen")
    try:
        palette_np =  palgen.run(image_path=image_path)
        if palette_np is None:
            logger.error(f"Could not generate a palette for {image_path}")
        one_hot_np = one_hot_encoding(palette_np, 256)

        hdf5['input'][idx] = img_np
        hdf5['target'][idx] = one_hot_np
        return 0
    except KeyError as e:
        logger.error(f'error inserting into dataset: {e}')
        return -1



def build_dataset(dataset_dir):

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('dataset creation')
    logger_palgen = logging.getLogger('Palgen')

    parent_dir = os.path.dirname(dataset_dir)

    palgen = PalGenWrapper(num_colors=64, num_jobs=8, logger=logger_palgen)

    train_dataset_path = os.path.join(dataset_dir, 'train2017')
    test_dataset_path = os.path.join(dataset_dir, 'test2017')
    val_dataset_path = os.path.join(dataset_dir, 'val2017')

    data_set_partitions = {'val':val_dataset_path, 'test':test_dataset_path}

    for _, (name, partition) in enumerate(data_set_partitions.items()):
        file_list = [file for file in os.listdir(partition) if os.path.isfile(os.path.join(partition, file))]
        logger.info(f"processing data in: {partition}")
        logger.info(f"Will add {len(file_list)} images to dataset file")
        logger.info(f"Creating hdf5 file for {name} set at {parent_dir}/{name}.hdf5")
        with h5py.File(os.path.join(parent_dir,f'{name}.hdf5'), "w") as hf:
            N = len(file_list)
            input_shape = (N, 3, 256, 256)
            target_shape = (N, 64, 3, 256)
            hf.create_dataset('input', shape=input_shape, dtype='float32', chunks=(1,3, 256, 256), compression='gzip')
            hf.create_dataset('target', shape=target_shape, dtype='uint8', chunks=(1,64, 3, 256), compression='gzip')
            idx = 0
            for  img in file_list:
                logger.info(f"{idx/len(file_list)}% complete")
                logger.info(f"Adding {img} to dataset")
                img_path = os.path.join(partition, img)
                rslt = add_data_hdf5(img_path, hf, logger_palgen, palgen, idx)

                if rslt == -1:
                    logger.error(f"Could not add {img} to dataset")
                else:
                    logger.info(f"Added {img} to dataset")
                    idx += 1


if __name__ == '__main__':
    dataset_source = 'dataset_source'
    build_dataset(dataset_source)
