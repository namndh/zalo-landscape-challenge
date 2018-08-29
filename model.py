import os 
import sys 
import torch 
import torchvision
from torch.autograd import Variable 
import torch.nn.functional as F 
import torch.nn as nn
import torch.optim as optim 
import torchvision.models as models

import constants

class CustomModel(nn.Module):
	def  __init__(self, pretrained_model, softmax, num_classes = constants.NUM_LABELS):
		super(CustomModel, self).__init__()
		self.pretrained_model = pretrained_model
		self.is_softmax = softmax
		self.fc = nn.Linear(1000, constants.NUM_LABELS)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		if self.is_softmax:
			return(self.softmax(self.fc(self.pretrained_model(x))))
		else:
			return(self.fc(self.pretrained_model(x)))


		