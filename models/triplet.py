import utils.models as models
import torch
import torch.nn as nn

def model_freeze(model):
	for param in model.parameters():
		param.requires_grad = False

	for param in classifier_model.fc.parameters():
		param.requires_grad = True
	for param in classifier_model.bn.parameters():
		param.requires_grad = True
	for param in classifier_model.logits.parameters():
		param.requires_grad = True

def model_finetune(model, ratio=0.3):
	model.base_model_finetune(ratio)

def model_init(device):
	model = models.BaseResNet18(num_classes=1)

	model.base_model_freeze()

	return model.to(device)
