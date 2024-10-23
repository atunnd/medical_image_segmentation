import tensorflow as tf
import os
import SimpleITK as sitk
import numpy as np
import shutil
import torch
import copy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from skimage.transform import resize
from transforms import train_transform, val_transform

class CustomDataset(tf.keras.utils.Sequence):

    def __init__(self, 
                 data_path,
                 n_train_patients=50, 
                 n_test_patients=20, 
                 task = 'La Cavity', 
                 transforms = None):

        super().__init__()
        self.data_path = data_path
        self.n_train_patients = n_train_patients
        self.n_test_patients = n_test_patients
        self.task = task
        self.transform = transforms

        if task == "La Cavity":
            self.training_files= os.listdir(os.path.join(self.data_path, "Training Set"))[:self.n_train_patients]
            self.testing_files= os.listdir(os.path.join(self.data_path, "Testing Set"))[:self.n_test_patients]
            self.input_format = 'lgemri.nrrd'
            self.output_format = 'laendo.nrrd'
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.training_files)
        elif self.mode == 'val':
            return len(self.testing_files)

    def load_nrrd(self, filename_path):
        data = sitk.ReadImage(filename_path)						
        data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
        data = sitk.GetArrayFromImage(data)								
        return (data)
    
    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.mode == "train":
            name = self.training_files[idx]
            img_array = self.load_nrrd(os.path.join(self.data_path, "Training Set", self.training_files[idx], self.input_format))
            label_array = self.load_nrrd(os.path.join(self.data_path, "Training Set", self.training_files[idx], self.output_format))
        
        elif self.mode == 'val':
            name = self.testing_files[idx]
            img_array = self.load_nrrd(os.path.join(self.data_path, "Testing Set", self.testing_files[idx], self.input_format))
            label_array = self.load_nrrd(os.path.join(self.data_path, "Testing Set", self.testing_files[idx], self.output_format))

        img_array = img_array.reshape(1, np.shape(img_array)[0], np.shape(img_array)[1], np.shape(img_array)[2])
        
        temp = np.empty(shape=[2, np.shape(label_array)[0], np.shape(label_array)[1], np.shape(label_array)[2]])
        temp[0, :, :, :] = 255 - label_array
        temp[1, :, :, :] = label_array
        label_array = np.reshape(temp, newshape=[-1, np.shape(label_array)[0], np.shape(label_array)[1], np.shape(label_array)[2]])
        
        if img_array.shape[2] == 576:
            padding = (32, 32, 32, 32)
            img_array = F.pad(torch.from_numpy(img_array), padding)
            label_array = F.pad(torch.from_numpy(label_array), padding)
            img_array = img_array.numpy()
            label_array = label_array.numpy()
        
        img_array = img_array[:, 19:83, :, :]
        label_array = label_array[:, 19:83, :, :]


        img_array= resize(img_array, (np.shape(img_array)[0], np.shape(img_array)[1], 128, 128), anti_aliasing=True)
        label_array= resize(label_array, (np.shape(label_array)[0], np.shape(label_array)[1], 128, 128), anti_aliasing=True)
        
        proccessed_out = {'name': name,
                          'image': img_array, 'label': label_array}
        if self.transform:
            if self.mode == "train":
                proccessed_out = self.transform[0](proccessed_out)
            elif self.mode == "val":
                proccessed_out = self.transform[1](proccessed_out)
            else:
                proccessed_out = self.transform(proccessed_out)
        return proccessed_out


def get_train_val_test_Dataloaders(data_path, train_batch_size, val_batch_size, num_workers):
    dataset = CustomDataset(data_path,
                            n_train_patients=80,
                            n_test_patients=20,
                            task = 'La Cavity',
                            transforms=[train_transform, val_transform])

    train_set, val_set= copy.deepcopy(dataset), copy.deepcopy(dataset)
    
    train_set.set_mode('train')
    val_set.set_mode('val')

    train_dataloader = DataLoader(
        dataset=train_set, batch_size=train_batch_size, shuffle=False, num_workers= num_workers)
    val_dataloader = DataLoader(
        dataset=val_set, batch_size=val_batch_size, shuffle=False, num_workers= num_workers)
    
    return train_dataloader, val_dataloader
