import os
import json
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm


class ImageCaptioner:
    def __init__(self):
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_caption(self, img_path, prompt):
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        description = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return description

def list_subdirectory_and_img_files(directory):
    subdirectories_and_img_files = {}

    for root, dirs, files in os.walk(directory):
        relative_path = os.path.relpath(root, directory)
        if relative_path != '.':
            subdirectories_and_img_files[relative_path] = []
            for file_name in files:
                if file_name.endswith(".png") or file_name.endswith(".jpg"):
                    subdirectories_and_img_files[relative_path].append(file_name)
    return subdirectories_and_img_files

def get_description_by_episode_name(data, episode_name):
    if episode_name in data:
        return data[episode_name]["description"]
    else:
        return None

def main(json_path, img_dir, output_json_path):
    output_json_dic = {}
    captioner = ImageCaptioner()
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as output_json_file:
            output_json_dic = json.load(output_json_file)
    else:
        output_json_dic = {}

    imgs = list_subdirectory_and_img_files(img_dir)
    split_episodes = list(imgs.keys())
    split_episodes.sort()


    for split_episode in tqdm(split_episodes):
        episode_path = os.path.join(img_dir, split_episode)
        episode = split_episode.split("/")[-1]

        if episode in data:
            expression = data[episode]["description"]
        else:
            continue

        for img in ["0", "1"]:
            key = f"{episode}/{img}"

            if key in output_json_dic:
                continue

            if os.path.exists(os.path.join(episode_path, f"{img}.png")):
                img_path = os.path.join(episode_path, f"{img}.png")
            elif os.path.exists(os.path.join(episode_path, f"{img}.jpg")):
                img_path = os.path.join(episode_path, f"{img}.jpg")

            prompt = f"""
            Give a clear, comprehensive and detailed description of the state of the objects shown in this image. For each object, mention their colors, sizes, shapes, how they are placed (upright, etc.), position within the image and relative position to other objects.
            Begin with the phrase 'In the image,'.
            Only use information that can be gained from the image.
            Mention the objects that appear in the sentence string below. If the objects in the sentence string are not present in the image, mention that they are not present.
            Sentence string: '{expression}'
            """
            response = captioner.generate_caption(img_path, prompt)
            output_json_dic[key] = response
            with open(output_json_path, "w") as output_json_file:
                json.dump(output_json_dic, output_json_file, indent=4)


if __name__ == "__main__":
    with open("configs/config.json") as config_file:
        config = json.load(config_file)
    img_dir = os.path.join("data", config["dataset_name"])
    if config["dataset_name"] == "SP-RT-1":
        json_path = os.path.join(img_dir, "info.json")
    elif config["dataset_name"] == "SP-HSR":
        json_path = os.path.join(img_dir, "hsr_info.json")
    output_json_dir = os.path.join(img_dir, "instructblip_embeddings")
    os.makedirs(output_json_dir, exist_ok=True)
    output_json_path = os.path.join(output_json_dir, "captions.json")
    main(json_path, img_dir, output_json_path)
