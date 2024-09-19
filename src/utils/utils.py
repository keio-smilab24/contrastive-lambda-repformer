import torch
import numpy as np
import random
import os
from datetime import datetime
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import loralib as lora
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

def get_seed_worker():
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed()

        # Seed other libraries with torch's seed
        random.seed(worker_seed)

        # Numpy seed must be between 0 and 2**32 - 1
        if worker_seed >= 2 ** 32:
            worker_seed = worker_seed % 2 ** 32
        np.random.seed(worker_seed)
    return seed_worker

def save_checkpoint(model, filename, adopt_lora=False):
    if adopt_lora:
        torch.save(model.state_dict(), filename)
        torch.save(lora.lora_state_dict(model), filename.replace(".pth", "_lora.pth"))
    else:
        torch.save(model.state_dict(), filename)

def load_checkpoint(model, filename, adopt_lora=False):
    if adopt_lora:
        model.load_state_dict(torch.load(filename), strict=False)
        model.load_state_dict(filename.replace(".pth", "_lora.pth"), strict=False)
    else:
        model.load_state_dict(torch.load(filename))
    model.eval()

def create_checkpoint_dir(base_dir="checkpoints"):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(base_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def find_trainable_layers(model):
    trainable_layers = {}
    non_trainable_layers = {}

    for name, child in model.named_children():
        if any(param.requires_grad for param in child.parameters()):
            trainable_layers[name] = child
        else:
            non_trainable_layers[name] = child

    print("Trainable Layers:")
    for name in trainable_layers.keys():
        print(f" - {name}")

    print("Non-Trainable Layers:")
    for name in non_trainable_layers.keys():
        print(f" - {name}")


def init_weights_he_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def text_to_ids(text, tokenizer, max_length):
        return tokenizer.encode(text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def plot_confusion_matrices(task_names, task_metrics, output_path):
    n_tasks = len(task_names)
    cols = int(np.ceil(np.sqrt(n_tasks)))
    rows = int(np.ceil(n_tasks / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if rows == 1 or cols == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.ravel()

    for ax, task_name in zip(axes, task_names):
        TP, FP, FN, task_total = task_metrics[task_name]
        TN = task_total - (TP + FP + FN)

        confusion_matrix = np.array([[TN, FP], [FN, TP]])

        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 24})
        ax.set_title(f'Task: {task_name}')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')

    for i in range(n_tasks, rows * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
