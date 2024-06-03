from torchvision.transforms import v2
from torchvision import transforms
import torch

INPUT_SIZE = (299, 299)
INPUT_SIZE_RESNET = (224, 224)

data_transforms = {
	"train": v2.Compose([
		v2.RandomResizedCrop(size=INPUT_SIZE, antialias=True),
		v2.RandomHorizontalFlip(p=0.5),
		v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
		v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=10),
		transforms.ToTensor(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	"val": v2.Compose([
		# v2.CenterCrop(size=INPUT_SIZE),
		v2.Resize(size=INPUT_SIZE),
		transforms.ToTensor(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	"verify": v2.Compose([
		# v2.CenterCrop(size=INPUT_SIZE),
		v2.Resize(size=INPUT_SIZE),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	"train_resnet": v2.Compose([
		v2.RandomResizedCrop(size=INPUT_SIZE_RESNET, antialias=True),
		v2.RandomHorizontalFlip(p=0.5),
		v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
		v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=10),
		# transforms.ToTensor(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	"val_resnet": v2.Compose([
		v2.Resize(size=INPUT_SIZE_RESNET),
		# transforms.ToTensor(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	"verify_resnet": v2.Compose([
		v2.Resize(size=INPUT_SIZE_RESNET),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

BASE_FOLDER = "/mnt/d/prjs/face_reco/11-785-fall-20-homework-2-part-2/face_cut_data/"
BATCH_SIZE = 16
