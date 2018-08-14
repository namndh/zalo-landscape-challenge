import os 
import sys
import numpy as np 
import h5py 
import cv2	
import matplotlib.pyplot as plt
import constants
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

HDF5_PATH = os.path.join(constants.DATA_DIR, 'trainval_data.hdf5')

hdf5_file = h5py.File(HDF5_PATH, "r")

data_num = hdf5_file["train_imgs"].shape[0]

print(type(hdf5_file["train_imgs"][0]))
img = np.reshape(hdf5_file["train_imgs"][0], (256, 256))
# imgplot = plt.imshow(img, cmap='gray')
# plt.show()

transform = transforms.Compose(
	[transforms.ToTensor(),
		transforms.Normalize(())
	])

train_set

for img, label in zip(hdf5_file["train_imgs"], hdf5_file["train_labels"]):


cv2.imshow('IMG', img)
cv2.waitKey(2000)
cv2.destroyAllWindows()

print(hdf5_file["train_imgs"][0].shape)


print(data_num)

class ZaloLandscapeDataset(Dataset):
	def __init__(self, hdf5_file, root_dir, train, transform=None):
		if os.path.isfile(hdf5_file):
			self.hdf5_file = h5py.File(hdf5_file)
			if train:
				self.train = self.hdf5_file["train_imgs"]
			else:
				self.
		
		
