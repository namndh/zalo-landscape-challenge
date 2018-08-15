import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os 
import h5py


class ZaloLandscapeDataset(Dataset):
	def __init__(self, hdf5_file, root_dir, train, transform=None):
		if os.path.isfile(hdf5_file):
			self.hdf5_file = h5py.File(hdf5_file)
			self.train = train
			if train:
				self.train_imgs = self.hdf5_file["train_imgs"]
				self.train_labels = self.hdf5_file["train_labels"]
			else:
				self.val_imgs = self.hdf5_file["val_imgs"]
				self.val_labels = self.hdf5_file["val_labels"]
			self.root_dir = root_dir
			self.transform = transform
		else:
			print('Data path is not available!')
			exit(1)

	def __len__(self):
		if self.train:
			return (len(self.train_imgs))
		else: 
			return (len(self.val_imgs))

	def __getitem__(self, idx):
		if self.train:
			image = self.train_imgs[idx, ...]
			label = self.train_labels[idx]
		else:
			image = self.val_imgs[idx, ...]
			label = self.val_imgs[idx]

		if self.transform:
			image = self.transform(image)

		return image, label
