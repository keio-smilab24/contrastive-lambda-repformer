import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import clip
from src.utils.utils import freeze_model, get_seed_worker
import h5py
from src.utils.multimodal_LLM import *
import time

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.initialize_models()
        self.load_config()
        self.load_data()

    def initialize_models(self):
        self.clip, self.clip_preprocessor = clip.load("RN101", device=self.device)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        freeze_model(self.clip)
        freeze_model(self.bert_model)
        self.clip_layer_hook = self.clip.visual.layer3.register_forward_hook(self.get_intermediate_output)

    def load_config(self):
        try:
            with open("configs/config.json") as config_file:
                config = json.load(config_file)
            self.dataset_name = config["dataset_name"]
        except FileNotFoundError:
            print("Config file not found. Using default settings.")

    def load_data(self):
        if self.dataset_name == "SP-RT-1":
            self.load_data_for_sprt1()
        elif self.dataset_name == "SP-HSR":
            self.load_data_for_sphsr()

    def create_or_open_hdf5(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return h5py.File(file_path, 'a')

    def get_intermediate_output(self, module, input, output):
        self.intermediate_output = (
            output  # torch.Size([batch_size*num_images, 1024, 14, 14])
        )

    def get_bert_emb(self, text, max_length):
        inputs = self.bert_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(self.device)
        return self.bert_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).pooler_output

    def save_embedding_to_hdf5(self, hdf5_file, path, data):
        if path in hdf5_file:
            del hdf5_file[path]
        hdf5_file.create_dataset(path, data=data.cpu().numpy())

    def load_embedding_from_hdf5(self, hdf5_file, path):
        if path in hdf5_file:
            data = hdf5_file[path][...]
            # if "scene_narrative/bert" in path:
            #     print(data.shape)
            return torch.tensor(data)
        else:
            raise KeyError(f"Path {path} does not exist in the HDF5 file.")

    def __getitem__(self, index):
        sample = self.data[index]
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        return sample["images"], sample["texts"], sample["image_paths"], label_tensor

    def __len__(self):
        return len(self.data)

    def load_or_compute_embedding(self, group_path, compute_embedding, hdf5_file):
        if f"data/SP-RT-1/instructblip_embeddings/" in group_path:
            data_0 = np.load(f"{group_path}_0.npz")["embeddings"]
            data_1 = np.load(f"{group_path}_1.npz")["embeddings"]
            stacked_data = np.concatenate([data_0, data_1])
            return torch.tensor(stacked_data)
        elif f"data/SP-HSR/instructblip_embeddings/" in group_path:
            data_0 = np.load(f"{group_path}_0.npz")["embeddings"]
            data_1 = np.load(f"{group_path}_1.npz")["embeddings"]
            stacked_data = np.concatenate([data_0, data_1])
            return torch.tensor(stacked_data)
        elif group_path in hdf5_file:
            return self.load_embedding_from_hdf5(hdf5_file, group_path)
        else:
            embedding = compute_embedding()
            self.save_embedding_to_hdf5(hdf5_file, group_path, embedding)
            return embedding

    def process_episode_images(self, image_paths):
        episode_images = [Image.open(img_path) for img_path in image_paths]
        stacked_episode_image = torch.stack([self.clip_preprocessor(image) for image in episode_images]).to(self.device)
        clip_image = self.clip.encode_image(stacked_episode_image).float()
        return clip_image

    def read_npz_file(self, directory, file_name):
        combined_data = np.array([np.load(os.path.join(directory, f"{file_name}_{i}.npz"))["arr_0"] for i in range(2)])

        return combined_data

    def get_label_ratio(self):
        count_dict = {"True": 0, "False": 0}
        for sample in self.data:
            label = sample["label"]
            if label:
                count_dict["True"] += 1
            else:
                count_dict["False"] += 1
        return count_dict

    def compute_scene_narratives(self, image_paths, root_key, model_type):
        scene_narratives = []
        for index in range(len(image_paths)):
            key = f"{root_key}/{index}"
            scene_narrative = get_scene_narrative_instruct_blip(key, f"data/{self.dataset_name}/instruct_blip.json")
            scene_narratives.append(scene_narrative)

        if model_type == 'bert':
            embeddings = [self.get_bert_emb(narrative, max_length=128) for narrative in scene_narratives]
        elif model_type == 'ada':
            embeddings = [torch.tensor(get_gpt3_embeddings(narrative)).unsqueeze(0) for narrative in scene_narratives]
            time.sleep(0.5)

        return torch.cat(embeddings, dim=0)

    def load_data_for_sprt1(self):
        with open(f"data/{self.dataset_name}/info.json") as f:
            json_file = json.load(f)
        episodes = sorted(os.listdir(f"{self.data_dir}"), key=lambda x: int(x[7:]))
        hdf5_file_path = f'data/{self.dataset_name}/embeddings.h5'

        with self.create_or_open_hdf5(hdf5_file_path) as hdf5_file:
            for episode in tqdm(episodes, total=len(episodes)):
                if episode not in json_file:
                    continue

                image_paths = [f"{self.data_dir}/{episode}/{img}" for img in sorted(os.listdir(f"{self.data_dir}/{episode}"))]
                if len(image_paths) != 2:
                    print(f"Error: {episode} has {len(image_paths)} images")
                    continue

                clip_image = self.load_or_compute_embedding(
                    f"clip/clip1d/{episode}",
                    lambda: self.process_episode_images(episode, image_paths),
                    hdf5_file)

                clip2d = self.load_or_compute_embedding(
                    f"clip/clip2d/{episode}",
                    lambda: self.intermediate_output.float(),
                    hdf5_file)

                vit_image = self.read_npz_file(
                    f"{os.path.dirname(self.data_dir)}/vit_embeddings",
                    f"{episode}"
                )

                dinov2_image = self.read_npz_file(
                    f"{os.path.dirname(self.data_dir)}/dino_embeddings",
                    f"{episode}"
                )

                bert_scene_narratives = self.load_or_compute_embedding(
                    f"data/SP-RT-1/instructblip_embeddings/bert/{episode}",
                    None,
                    None)

                ada_scene_narratives = self.load_or_compute_embedding(
                    f"data/SP-RT-1/images/instructblip_embeddings/ada/{episode}",
                    None,
                    None)

                # Instructions Embeddings
                inst = json_file[episode]['description']

                bert_inst = self.load_or_compute_embedding(
                    f"instruction/bert/{episode}",
                    lambda: self.get_bert_emb(inst, 16).squeeze(0),
                    hdf5_file)

                clip_inst = self.load_or_compute_embedding(
                    f"instruction/clip/{episode}",
                    lambda: self.clip.encode_text(clip.tokenize(inst).to(self.device)).squeeze(0).float(),
                    hdf5_file)

                ada_inst = self.load_or_compute_embedding(
                    f"instruction/ada/{episode}",
                    lambda: torch.tensor(get_gpt3_embeddings(str(inst))),
                    hdf5_file)

                self.data.append({
                    "images": {"clip2d_images": clip2d, "clip_images": clip_image, "vit_images": vit_image, "dinov2_images": dinov2_image, "bert_scene_narratives": bert_scene_narratives.to(self.device), "ada_scene_narratives": ada_scene_narratives.to(self.device)},
                    "image_paths": image_paths,
                    "texts": {"bert": bert_inst.to(self.device), "clip": clip_inst.to(self.device), "ada": ada_inst.to(self.device)},
                    "label": json_file[episode]["succeeded"]
                })

    def load_data_for_sphsr(self):
        with open(f"data/{self.dataset_name}/hsr_info.json") as f:
            json_file = json.load(f)
        # print(self.data_dir)
        episodes = sorted(os.listdir(f"{self.data_dir}"), key=lambda x: int(x[11:]))
        hdf5_file_path = f'data/{self.dataset_name}/embeddings.h5'

        with self.create_or_open_hdf5(hdf5_file_path) as hdf5_file:
            for episode in tqdm(episodes, total=len(episodes)):
                print("episode", episode)
                if episode not in json_file:
                    continue

                image_paths = [f"{self.data_dir}/{episode}/{img}" for img in sorted(os.listdir(f"{self.data_dir}/{episode}"))]
                if len(image_paths) != 2:
                    print(f"Error: {episode} has {len(image_paths)} images")
                    continue

                clip_image = self.load_or_compute_embedding(
                    f"clip/clip1d/{episode}",
                    lambda: self.process_episode_images(episode, image_paths),
                    hdf5_file)

                clip2d = self.load_or_compute_embedding(
                    f"clip/clip2d/{episode}",
                    lambda: self.intermediate_output.float(),
                    hdf5_file)

                vit_image = self.read_npz_file(
                    f"{os.path.dirname(self.data_dir)}/vit_embeddings",
                    f"{episode}"
                )

                dinov2_image = self.read_npz_file(
                    f"{os.path.dirname(self.data_dir)}/dino_embeddings",
                    f"{episode}"
                )


                bert_scene_narratives = self.load_or_compute_embedding(
                    f"data/SP-HSR/instructblip_embeddings/bert/{episode}",
                    None,
                    None)


                ada_scene_narratives = self.load_or_compute_embedding(
                    f"data/SP-HSR/instructblip_embeddings/ada/{episode}",
                    None,
                    None)

                # Instructions Embeddings
                inst = json_file[episode]['description']

                bert_inst = self.load_or_compute_embedding(
                    f"instruction/bert/{episode}",
                    lambda: self.get_bert_emb(inst, 16).squeeze(0),
                    hdf5_file)

                clip_inst = self.load_or_compute_embedding(
                    f"instruction/clip/{episode}",
                    lambda: self.clip.encode_text(clip.tokenize(inst).to(self.device)).squeeze(0).float(),
                    hdf5_file)

                ada_inst = self.load_or_compute_embedding(
                    f"instruction/ada/{episode}",
                    lambda: torch.tensor(get_gpt3_embeddings(str(inst))),
                    hdf5_file)

                self.data.append({
                    "images": {"clip2d_images": clip2d, "clip_images": clip_image, "vit_images": vit_image, "dinov2_images": dinov2_image, "bert_scene_narratives": bert_scene_narratives.to(self.device), "ada_scene_narratives": ada_scene_narratives.to(self.device)},
                    "image_paths": image_paths,
                    "texts": {"bert": bert_inst.to(self.device), "clip": clip_inst.to(self.device), "ada": ada_inst.to(self.device)},
                    "label": json_file[episode]["succeeded"]
                })

def create_data_loaders(train_set, valid_set, test_set, batch_size, seed=42):
    seed_worker = get_seed_worker()
    g = torch.Generator()
    if seed != False:
        g.manual_seed(seed)
    # g.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    return train_loader, valid_loader, test_loader
