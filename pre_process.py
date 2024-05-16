import builtins
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import tqdm
import asyncio
import os
import scandir

# Paths
input_base_dir = '/mnt/d/prjs/face_reco/11-785-fall-20-homework-2-part-2/'
output_base_dir = '/mnt/d/prjs/face_reco/11-785-fall-20-homework-2-part-2/face_cut_data/'

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
	keep_all=True,
	device=device,
)

failed_images = open("failed_images.txt", "w")

async def process_and_save_image(input_path, output_path, pbar):
	try:
		# Load image
		image = Image.open(input_path).convert('RGB')
		width, height = image.size

		# Detect face
		boxes, _ = mtcnn.detect(image)
		if boxes is not None:
			boxes = [
				[max(0, left), max(0, top), min(width, right), min(height, bottom)]
				for left, top, right, bottom in boxes
			]

			boxes = sorted(boxes, reverse=True, key=lambda box: (box[0] - box[2]) * (box[1] - box[3]))
			for box in boxes:
				# Crop face from image
				left, top, right, bottom = box
				face = image.crop((left, top, right, bottom))

				output_dir = os.path.dirname(output_path)
				os.makedirs(output_dir, exist_ok=True)

				# Save processed image
				face.save(output_path)
				break  # Only save the first detected face
		else:
			failed_images.write(input_path + "\n")
	except Exception as e:
		print(f"Error processing {input_path}: {e}")
	finally:
		pbar.update(1)

def walk_dir(dir):
	for entry in scandir.scandir(dir):
		if entry.is_dir():
			for file in scandir.scandir(entry.path):
				if file.is_file():
					yield file.path, f"{entry.name}/{file.name}"
		elif entry.is_file():
			yield entry.path, entry.name


async def process_directory(input_dir, output_dir, num_workers):
	# Ensure the output directory exists
	os.makedirs(output_dir, exist_ok=True)

	tasks = []

	# Traverse the input directory with progress bar
	pbar = tqdm.tqdm(desc=f"Processing {input_dir.split('/')[-1]}", unit="file")

	checked = set()

	total_files = 0
	for input_path, entry_path in tqdm.tqdm(walk_dir(input_dir)):
		total_files += 1
		tasks.append(asyncio.create_task(process_and_save_image(input_path, os.path.join(output_dir, entry_path), pbar)))

	pbar.total = total_files
	pbar.refresh()

	await asyncio.gather(*tasks)

	pbar.close()

# Process train, test, and val directories
async def main(folder_type, sem):
	global input_base_dir
	global output_base_dir
	input_base_dir = os.path.join(input_base_dir, folder_type)
	output_base_dir = os.path.join(output_base_dir, folder_type)

	match folder_type:
		case "classification_data":
			sub_folders =  [
				"val_data",
				"train_data",
				"test_data"
			]
			for split in sub_folders:
				input_split_dir = os.path.join(input_base_dir, split)
				output_split_dir = os.path.join(output_base_dir, split)
				await process_directory(input_split_dir, output_split_dir, sem)
		case "verification_data":
			await process_directory(input_base_dir, output_base_dir, sem)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		prog="Pre-Process Dath For Face Recognition",
		description="Cut the Face of the given data",
		epilog="Pre-Process between `classification_data` and `verification_data`",
	)

	parser.add_argument("folder_type", choices=["classification_data", "verification_data"])
	parser.add_argument("-w", "--workers", type=int, default=4)

	args = parser.parse_args()

	asyncio.run(main(args.folder_type, args.workers))

failed_images.close()
print("Processing complete.")
