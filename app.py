import gradio as gr
import faiss
import numpy as np

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from config import data_transforms

import timm
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dim = 512

model = InceptionResnetV1(pretrained="vggface2", device=device)
mtcnn = MTCNN(device=device, image_size=224)
model.eval()

spoof = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
spoof.head = torch.nn.Linear(spoof.head.in_features, 2)
spoof = spoof.to(device)
spoof.load_state_dict(torch.load("spoof.pth"))
spoof.eval()

names_file = "names_list.npy"
index_file = "faiss_index.bin"

try:
	index = faiss.read_index(index_file)
except:
	index = faiss.IndexFlatIP(dim)

try:
	names_list = np.load(names_file).tolist()
except:
	names_list = []

def detect_face(image):
	global mtcnn

	img = Image.fromarray(image)
	face = mtcnn(img, save_path="_tmp.png")
	if face is not None:
		return face.unsqueeze(0).to(device), "_tmp.png"

	return None, None

def register_face(image, name):
	global model
	global index
	global names_list

	face_tensor, face_img = detect_face(image)
	if face_tensor is None:
		return "No face detected, cannot register."

	# Extract face embedding
	with torch.no_grad():
		embedding = model(face_tensor).cpu().numpy()

	# Normalize embedding for cosine similarity
	embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
	
	# Add embedding to FAISS index
	index.add(embedding)

	names_list.append(name)
	
	# Store face_id
	return f"Face registered with name: {name}"

def process_image(image, threshold):
	global model
	global index
	global names_list

	face_tensor, face_img = detect_face(image)

	if face_tensor is None:
		return "No face detected."

	# Extract face embedding
	with torch.no_grad():
		embedding = model(face_tensor).cpu().numpy()
		check_spoof = spoof(face_tensor).cpu()
		check_spoof = check_spoof.argmax(dim=1, keepdim=True)

	msg = ""

	if check_spoof.eq(1):
		msg += "This picture is not spoofed. "
	else:
		msg += "This picture is spoofed. "

	# Normalize embedding for cosine similarity
	embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
	
	# Search for similar faces in the index
	distances, indices = index.search(embedding, 1)

	# print(distances[0][0], threshold)
	if distances[0][0] < threshold:
		msg += "Face not recognized. Please register."
	else:
		# print(names_list)
		name = names_list[indices[0][0]]
		msg += f"Face recognized! Name: {name}"

	print(msg)
	return msg, face_img

def registration(image, name):
	face_tensor, face_img = detect_face(image)
	if face_tensor is None:
		return "No face detected, cannot register."

	return register_face(image, name), face_img

def save_index():
	global index
	global names_list

	faiss.write_index(index, index_file)
	np.save(names_file, np.array(names_list))
	return "FAISS index saved."

# Gradio interface
with gr.Blocks() as demo:
	with gr.Row():
		with gr.Column():
			image_input = gr.Image(sources=["upload"], label="Upload Image")
			recognition_button = gr.Button("Recognize Face")
			registration_button = gr.Button("Register Face")
			output_text = gr.Textbox(label="Output")
			face_output = gr.Image(label="Detected Face", type="filepath")

		with gr.Column():
			name_input = gr.Textbox(label="Name")
			threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.6, label="Threshold")
			save_index_button = gr.Button("Save FAISS Index")

	recognition_button.click(process_image, inputs=[image_input, threshold_slider], outputs=[output_text, face_output])
	registration_button.click(registration, inputs=[image_input, name_input], outputs=[output_text, face_output])
	save_index_button.click(save_index, inputs=None, outputs=output_text)

# Launch the Gradio interface
demo.launch()
