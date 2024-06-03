import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseResNet18(nn.Module):
	def __init__(self, embedding_dim=128, classify=False, num_classes=4000):
		super().__init__()

		self.classify = classify

		self.base_model = models.resnet18(pretrained=True)

		num_features_in = self.base_model.fc.in_features
		self.base_model.fc = nn.Linear(num_features_in, embedding_dim)
		self.bn = nn.BatchNorm1d(embedding_dim)
		self.logits = nn.Linear(embedding_dim, num_classes)

		self.base_model_freeze()

	def forward(self, x):
		x = self.base_model(x)
		x = self.bn(x)
		if self.classify:
			x = self.logits(x)
		else:
			x = F.normalize(x, p=2, dim=1)
		return x

	def base_model_freeze(self):
		for param in self.base_model.parameters():
			param.requires_grad = False
			
		for param in self.base_model.fc.parameters():
			param.requires_grad = True

	def base_model_finetune(self, ratio=0.3):
		params = list(self.base_model.parameters())
		for param in params[-int(len(params) * ratio):]:
			param.requires_grad = True

