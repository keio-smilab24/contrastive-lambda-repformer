import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, ViTModel

# Load ViT model and image processor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model.eval()

data_type = "RT-1"

if data_type == "RT-1":
    data_dir = "data/RT-1/images"
elif data_type == "HSR":
    data_dir = "data/zero-shot/images"

output_dir = os.path.join(data_dir, "image_embeddings")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

if data_type == "RT-1":
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(data_dir, split)

        for episode_dir in tqdm(os.listdir(split_dir), desc=f"Processing {split}"):

            episode_path = os.path.join(split_dir, episode_dir)

            for img_file in os.listdir(episode_path):
                img_path = os.path.join(episode_path, img_file)

                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img)
                    # print(outputs.last_hidden_state.shape)
                    features = outputs.last_hidden_state[0][0]

                episode_name = os.path.splitext(img_file)[0]
                feature_file_name = f"{episode_dir}_{episode_name}"

                # print(features.shape)
                os.makedirs(output_dir, exist_ok=True)
                np.savez(os.path.join(output_dir, feature_file_name), features.squeeze().numpy())

print(f"Feature size: {features.squeeze().numpy().shape}")
