import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import argparse
import pickle
from os import path

from config import BASE_FOLDER, BATCH_SIZE, data_transforms
from utils.train import fit, train_classifier, train_triplet
import utils.loss as loss
from dataloader import FaceDataset

import models.classifier as classifier
import models.triplet as triplet
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser(description="Train CNN models for Cifar10.")
parser.add_argument(
	"-r", "--resume",
	action='store_true',
	help="resume previous training"
)
parser.add_argument(
	"-n", "--epochs",
	type=int, default=10,
	help="number of epochs to train"
)
parser.add_argument(
	'method',
	type=str, choices=["classifier", "triplet"],
	help="method of training"
)
parser.add_argument(
	"-s", "--step",
	type=str, choices=["fe", "ft"], default="fe",
	help="Step of Transfer Learning"
)
args = parser.parse_args()


try:
	f = open("/mnt/d/prjs/face_reco/face_reco.pickle", "rb+")
except IOError:
	try:
		f = open("/mnt/d/prjs/face_reco/face_reco.pickle", "wb+")
	except IOError:
		print("Can't open pickle _jar_ for training")
		sys.exit()

try:
	checkpoints = pickle.load(f)
except EOFError:
	checkpoints = {
		"epochs": [],
		"best": {},
		"history": [], # { "val": { "acc": [], "loss": [] }, "train": { "acc": [], "loss": [], }	}
	}

f.close()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Loading dataset
match args.method:
	case "classifier":
		val_ds = ImageFolder(root=path.join(BASE_FOLDER, "classification_data/val_data"), transform=data_transforms["val"])
		test_ds = ImageFolder(root=path.join(BASE_FOLDER, "classification_data/test_data"), transform=data_transforms["val"])
		train_ds = ImageFolder(root=path.join(BASE_FOLDER, "classification_data/train_data"), transform=data_transforms["train"])
		print(len(train_ds.classes))

		val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True) 
		test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True) 
		train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) 

		assert len(train_ds.classes) == len(val_ds.classes) == len(test_ds.classes)
		num_classes = len(train_ds.classes)

		model = classifier.model_init(device, num_classes)
		if args.step == "ft":
			classifier.model_fine_tune(model)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
		train_fn = train_classifier

	case "triplet":
		val_ds = FaceDataset(root=path.join(BASE_FOLDER, "classification_data/val_data"), transform=data_transforms["val_resnet"])
		# test_ds = FaceDataset(root=path.join(BASE_FOLDER, "classification_data/test_data"), transform=data_transforms["val"])
		train_ds = FaceDataset(root=path.join(BASE_FOLDER, "classification_data/train_data"), transform=data_transforms["train_resnet"])

		val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True) 
		# test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True) 
		train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) 

		assert len(train_ds.labels) == len(val_ds.labels)

		model = triplet.model_init(device)
		if args.step == "ft":
			triplet.model_fine_tune(model)

		criterion = loss.TripletMarginLossCosine()
		optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
		train_fn = train_triplet

start_epoch = 0
if args.resume:
	start_epoch = len(checkpoints["epochs"])
	model.load_state_dict(checkpoints["epochs"][start_epoch - 1]["model"])
	optimizer.load_state_dict(checkpoints["epochs"][start_epoch - 1]["optimizer"])

fit(
	f"{args.method}_{args.step}", model, train_fn,
	optimizer, criterion,
	train_dl, val_dl, device,
	checkpoints, start_epoch, start_epoch + args.epochs,
)

checkpoints["history"] = checkpoints["history"][:start_epoch + args.epochs]
checkpoints["epochs"] = checkpoints["epochs"][:start_epoch + args.epochs]
checkpoints["best"] = model.state_dict()

try:
	f = open("/mnt/d/prjs/cifar10.pickle", "wb+")
except IOError as e:
	print("Can't open pickle _jar_ for training")
	sys.exit()

with f:
	pickle.dump(checkpoints, f)

	f.close()
