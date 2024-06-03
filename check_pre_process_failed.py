from utils.utils import walk_dir
from tqdm import tqdm
import os

input_base_dir = '/mnt/d/prjs/face_reco/11-785-fall-20-homework-2-part-2/'
output_base_dir = '/mnt/d/prjs/face_reco/11-785-fall-20-homework-2-part-2/face_cut_data/'

if __name__ == "__main__":
	folders = [
		"verification_data",
		"classification_data/test_data",
		"classification_data/val_data",
		"classification_data/train_data",
	]
	for folder in folders:
		print(f"Checking {folder}:")
		total = 0
		missing = 0
		for input_path, entry_path in tqdm(walk_dir(f"{input_base_dir}/{folder}")):
			total += 1
			if not os.path.exists(f"{output_base_dir}/{folder}/{entry_path}"):
				missing += 1
		print(f"miss {missing / total * 100}% - {missing} files")
			
