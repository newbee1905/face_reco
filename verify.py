import argparse
from os import path
import numpy as np

from config import BASE_FOLDER, data_transforms
from dataloader import VerificationDataset

from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

import models.classifier as classifier

from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

parser = argparse.ArgumentParser(description="Train CNN models for Cifar10.")

parser.add_argument(
	'method',
	type=str, choices=["l2", "cos"],
	help="method of training"
)
parser.add_argument(
	"-t", "--threshold",
	nargs="+",
	type=float, default=[0.7, 0.7, 0.7],
	help="Threshold for face embedded similairty"
)

args = parser.parse_args()

while len(args.threshold) < 3:
	args.threshold.append(0.7)

# TODO: Load best model from pickle file
# try:
# 	f = open("/mnt/d/prjs/face_reco/face_reco.pickle", "rb+")
# except IOError:
# 	print("Can't open pickle _jar_ for training")
# 	sys.exit()


val_ds = VerificationDataset(
	BASE_FOLDER,
	path.join(BASE_FOLDER, "verification_pairs_val.txt"),
	transform=data_transforms["verify"],
)
val_dl = DataLoader(val_ds)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

base_model = InceptionResnetV1(pretrained="vggface2")
base_model.to(device)
base_model.eval()

classifier_model = classifier.model_init(device, 4000)
classifier_model.classify = False
classifier_model.load_state_dict(torch.load("/mnt/d/Downloads/model_3.pth"))
classifier_model.to(device)
classifier_model.eval()

y_vals = []
y_hats = {
	"base": [],
	"classifier": [],
}
scores = {
	"base": [],
	"classifier": [],
}

def emb_dist(method, model, img1, img2, threshold):
	emb1 = model(img1)
	emb2 = model(img2)

	match method:
		case "l2":
			dist = F.cosine_similarity(emb1, emb2)
			y_hat = (dist < threshold).item()
		case "cos":
			dist = F.cosine_similarity(emb1, emb2)
			y_hat = (dist > threshold).item()

	return dist.item(), y_hat

 
for i, (img1, img2, y_val) in enumerate(tqdm(val_dl)):
	if len(img1) == 0 or len(img2) == 0:
		continue	

	img1 = img1.to(device)
	img2 = img2.to(device)

	y_val = y_val.item()
	y_vals.append(y_val)

	base_score, base_y_hat = emb_dist(args.method, base_model, img1, img2, args.threshold[0])
	scores["base"].append(base_score)
	y_hats["base"].append(base_y_hat)

	classifier_score, classifier_y_hat = emb_dist(args.method, classifier_model, img1, img2, args.threshold[1])
	scores["classifier"].append(classifier_score)
	y_hats["classifier"].append(classifier_y_hat)

base_fpr, base_tpr, thresholds = roc_curve(y_vals, scores["base"])
base_auc = roc_auc_score(y_vals, scores["base"])
base_acc = accuracy_score(y_vals, y_hats["base"])

classifier_fpr, classifier_tpr, thresholds = roc_curve(y_vals, scores["classifier"])
classifier_auc = roc_auc_score(y_vals, scores["classifier"])
classifier_acc = accuracy_score(y_vals, y_hats["classifier"])

print(f"Base Model AUC: {base_auc}")
print(f"Base Model ACC (threshold: {args.threshold[0]}): {base_acc}")

print(f"Classifier Model AUC: {classifier_auc}")
print(f"Classifier Model ACC (threshold: {args.threshold[1]}): {classifier_acc}")

plt.plot(base_fpr, base_tpr, label='Base Model: ROC curve (area = %0.2f)' % base_auc)
plt.plot(classifier_fpr, classifier_tpr, label='Classifier Model: ROC curve (area = %0.2f)' % classifier_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
