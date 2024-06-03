import torch 
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from torchvision import models

from torchvision.datasets.utils import download_url
import tarfile
import os
from os import path

import pandas as pd
import scandir

import random

from config import BASE_FOLDER

class FaceDataset(Dataset):
	def __init__(self, root, transform=None, target_transform=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		# Mapping from label IDs to list of image paths
		self.label_to_images = {}
		self.labels = os.listdir(self.root)
		self.num_classes = len(self.labels)

		# Build the mapping
		for label in self.labels:
			folder_path = os.path.join(self.root, label)
			images_list = os.listdir(folder_path)
			images_list_abs = [os.path.join(folder_path, image_path) for image_path in images_list]
			self.label_to_images[label] = images_list_abs

		# All unique labels for negative sampling
		self.all_labels = list(self.label_to_images.keys())

		# Track index for each label
		self.label_indices = {label: 0 for label in self.all_labels}

	def __load_image(self, path):
		image = read_image(path)
		if self.transform:
			image = self.transform(image)
		return image

	def __getitem__(self, index):
		# Select anchor label sequentially
		anchor_label = self.all_labels[index % self.num_classes]
		anchor_images = self.label_to_images[anchor_label]

		# Select anchor image sequentially
		anchor_index = self.label_indices[anchor_label]
		anchor_image_path = anchor_images[anchor_index]

		# Update the index for next access
		self.label_indices[anchor_label] = (anchor_index + 1) % len(anchor_images)

		# Select positive image randomly from the same label
		positive_image_path = random.choice([img for img in anchor_images if img != anchor_image_path])

		# Select negative label and image randomly
		negative_label = random.choice([label for label in self.all_labels if label != anchor_label])
		negative_image_path = random.choice(self.label_to_images[negative_label])

		# Load images
		anchor_image = self.__load_image(anchor_image_path)
		positive_image = self.__load_image(positive_image_path)
		negative_image = self.__load_image(negative_image_path)

		# Apply target transform if any
		if self.target_transform:
			anchor_label = self.target_transform(anchor_label)
			negative_label = self.target_transform(negative_label)

		return anchor_image, positive_image, negative_image

	def __len__(self):
		return len(self.labels) * max(len(images) for images in self.label_to_images.values())


class VerificationDataset(torch.utils.data.Dataset):
	def __init__(self, image_dir, labels_file, type="val", transform=None):
		self.image_dir = image_dir
		self.labels_file = labels_file
		self.transform = transform
		self.type = type

		# Load pairs and labels from the text file
		self.pairs = []
		self.labels = []
		with open(labels_file, 'r') as f:
			for line in f:
				if type == "val":
					img1, img2, label = line.strip().split(' ')
					img1 = os.path.join(image_dir, img1)
					img2 = os.path.join(image_dir, img2)
					self.pairs.append((img1, img2))
					self.labels.append(int(label) == 1)
				elif type == "test":
					img1, img2 = line.strip().split(' ')
					img1 = os.path.join(image_dir, img1)
					img2 = os.path.join(image_dir, img2)
					self.pairs.append((img1, img2))

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		try:
			img1_path, img2_path = self.pairs[idx]
			img1 = self.__load_image(img1_path)
			img2 = self.__load_image(img2_path)
		except Exception as e:
			img1 = []
			img2 = []
			label = 0

		if self.type == "val":
			label = torch.tensor(self.labels[idx])
			return img1, img2, label
		elif self.type == "test":
			return img1, img2, img1_path, img2_path

	def __load_image(self, path):
		image = read_image(path)

		if self.transform:
			image = self.transform(image)

		return image
