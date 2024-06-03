from tqdm import tqdm
from tempfile import TemporaryDirectory
from os import path
import torch.nn.functional as F

import torch
import time

def train_classifier(phase, model, optimizer, criterion, dl, device):
	if phase == 'train':
		model.train()
	else:
		model.eval()

	running_loss = 0.0
	running_corrects = 0
	
	for i, data in enumerate(tqdm(dl)):
		X, y = data[0].to(device), data[1].to(device)
		optimizer.zero_grad()
		
		with torch.set_grad_enabled(phase == 'train'):
			y_hat = model(X)
			loss = criterion(y_hat, y)
			y_hat = torch.argmax(y_hat, dim=1)
			
			if phase == 'train':
				loss.backward()
				optimizer.step()

			running_loss += loss.item() * X.size(0)
			running_corrects += torch.sum(y_hat == y.data)

	ds_size = len(dl.dataset)
	epoch_loss = running_loss / ds_size
	epoch_acc = running_corrects.double() / ds_size
		
	print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
	return epoch_loss, epoch_acc

def train_triplet(phase, model, optimizer, criterion, dl, device):
	if phase == 'train':
		model.train()
	else:
		model.eval()

	running_loss = 0.0
	running_corrects = 0
	
	for i, data in enumerate(tqdm(dl)):
		anchor, pos, neg = data[0].to(device), data[1].to(device), data[2].to(device)
		optimizer.zero_grad()
		
		with torch.set_grad_enabled(phase == 'train'):
			anchor_hat = model(anchor)
			pos_hat = model(pos)
			neg_hat = model(neg)
			loss = criterion(anchor_hat, pos_hat, neg_hat)

			if phase == 'train':
				loss.backward()
				optimizer.step()

			running_loss += loss.item() * anchor.size(0)

			pos_sim = F.cosine_similarity(anchor_hat, pos_hat)
			neg_sim = F.cosine_similarity(anchor_hat, neg_hat)
			running_corrects += ((pos_sim - neg_sim) > criterion.margin).sum().item()

	ds_size = len(dl.dataset)
	epoch_loss = running_loss / ds_size
	epoch_acc = running_corrects.double() / ds_size
		
	print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
	return epoch_loss, epoch_acc

def fit(model_name, model, train_fn, optimizer, criterion, train_dl, val_dl, device, checkpoints, start_epoch, num_epochs):
	with TemporaryDirectory() as tempdir:
		since = time.time()

		best_model_params_path = path.join(tempdir, f"_model_{model_name}.pth")
		torch.save(model.state_dict(), best_model_params_path)

		best_acc = 0.0

		while start_epoch < num_epochs:
			print(f"Epoch {start_epoch + 1}/{num_epochs}")
			
			train_loss, train_acc = train_fn("train", model, optimizer, criterion, train_dl, device)
			val_loss, val_acc = train_fn("val", model, optimizer, criterion, val_dl, device)

			history = {
				"val": {
					"acc": val_acc.item(),
					"loss": val_loss,
				},
				"train": {
					"acc": train_acc.item(),
					"loss": train_loss,
				},
			}

			if val_acc > best_acc:
				best_acc = val_acc 
				torch.save(model.state_dict(), best_model_params_path)

			epoch_model = {
				"model": model.state_dict(),
				"optimizer": optimizer.state_dict(),
			}

			torch.save(model.state_dict(), f"/mnt/d/prjs/face_reco/trained/model_{model_name}_{start_epoch}.pth")
			torch.save(model.state_dict(), f"/mnt/d/prjs/face_reco/trained/optimizer_{model_name}_{start_epoch}.pth")

			if start_epoch == len(checkpoints["history"]):
				checkpoints["history"].append(history)
				checkpoints["epochs"].append(epoch_model)
			else:
				checkpoints["history"][start_epoch] = history
				checkpoints["epochs"][start_epoch] = epoch_model
			start_epoch += 1

		model.load_state_dict(torch.load(best_model_params_path))

		time_elapsed = time.time() - since
		print(f'Complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
		print(f'Best val Acc: {best_acc:4f}')
