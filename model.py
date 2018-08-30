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
	def  __init__(self, pretrained_model, num_classes = constants.NUM_LABELS):
		super(CustomModel, self).__init__()
		self.pretrained_model = pretrained_model

		self.num_ftrs = self.pretrained_model.fc.in_features

		self.shared = nn.Sequential(*list(self.pretrained_model.children())[:-1])
		self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

	def forward(self, x):
		x = self.shared(x)
		x = torch.squeeze(x)
		return self.target(x)

	def frozen_until(self, to_layer):
		print('Frozen pretrained model to {}-th layer'.format(to_layer))

		child_counter = 0
		for child in self.shared.children():
			if child_counter <= to_layer:
				print("child", child_counter, " was frozen")
				for param in child.parameters():
					param.require_grad = False
			else:
				print("child", child_counter, " was not frozen")
				for param in child.param.parameters():
					param.require_grad = True 
			child_counter += 1


def net_frozen(args, model):
	print('---------------------------------------------------------------')
	model.frozen_until(args.frozen)
	init_lr = args.lr 
	if args.optim == 'adam':
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = init_lr, weight_decay=args.weight_decay)
	elif args.optim == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=args.weight_decay, momentum=0.9)
	print('---------------------------------------------------------------')
	return model, optimizer
