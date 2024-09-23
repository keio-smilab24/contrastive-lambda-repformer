import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import requests

# Load DINOv2 model and image processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')
model.eval()

with open("configs/config.json") as config_file:
    config = json.load(config_file)
data_type = config["dataset_name"]

if data_type == "SP-RT-1":
    data_dir = "data/SP-RT-1/"
elif data_type == "SP-HSR":
    data_dir = "data/SP-HSR/"

output_dir = os.path.join(data_dir, "dino_embeddings")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

if data_type == "SP-RT-1":
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(data_dir, split)

        for episode_dir in tqdm(os.listdir(split_dir), desc=f"Processing {split}"):

            episode_path = os.path.join(split_dir, episode_dir)

            for img_file in os.listdir(episode_path):
                img_path = os.path.join(episode_path, img_file)

                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state[0][0]
                    # print(features.shape) # torch.Size([1, 257, 1024])

                episode_name = os.path.splitext(img_file)[0]
                feature_file_name = f"{episode_dir}_{episode_name}"

                os.makedirs(output_dir, exist_ok=True)
                np.savez(os.path.join(output_dir, feature_file_name), features.squeeze().numpy())

elif data_type == "SP-HSR":
    split = "hsr_data"
    split_dir = os.path.join(data_dir, split)
    for episode_dir in tqdm(os.listdir(split_dir), desc=f"Processing {split}"):

        episode_path = os.path.join(split_dir, episode_dir)

        for img_file in os.listdir(episode_path):
            img_path = os.path.join(episode_path, img_file)

            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[0][0]
                # print(features.shape) # torch.Size([1, 257, 1024])

            episode_name = os.path.splitext(img_file)[0]
            feature_file_name = f"{episode_dir}_{episode_name}"

            os.makedirs(output_dir, exist_ok=True)
            np.savez(os.path.join(output_dir, feature_file_name), features.squeeze().numpy())

print(f"Feature size: {features.squeeze().numpy().shape}")
