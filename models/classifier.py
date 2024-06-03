import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

def model_freeze(model):
	for param in model.parameters():
		param.requires_grad = False
	
	for param in classifier_model.last_linear.parameters():
		param.requires_grad = True
	for param in classifier_model.last_bn.parameters():
		param.requires_grad = True
	for param in classifier_model.logits.parameters():
		param.requires_grad = True

def model_finetune(model, ratio=0.3):
	params = list(model.parameters())
	for param in params[-int(len(params) * ratio):]:
		param.requires_grad = True

def model_init(device, num_classes):
	model = InceptionResnetV1(pretrained="vggface2", classify=True, device=device)

	embedding_dim = classifier_model.logits.in_features

	model.logits = nn.Linear(embedding_dim, num_classes)

	model_freeze(model)

	return model.to(device)
