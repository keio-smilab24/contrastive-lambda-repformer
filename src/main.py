import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import wandb
from datetime import datetime
from torchinfo import summary

from model import LambdaRepformer
from utils.data_loader import CustomDataset, create_data_loaders
from utils.utils import torch_fix_seed, create_checkpoint_dir, save_checkpoint, load_checkpoint
from train_model import train_model
from validate_model import validate_model
from test_model import test_model

def main():
    # Load configuration
    with open("configs/config.json") as config_file:
        config = json.load(config_file)

    # Set random seed
    if config["seed"] != False:
        torch_fix_seed(config["seed"])
    # torch_fix_seed(config["seed"])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare datasets
    data_path = f"{config['data_path']}/{config['dataset_name']}"
    train_set = CustomDataset(f"{data_path}/train")
    valid_set = CustomDataset(f"{data_path}/valid")
    test_set = CustomDataset(f"{data_path}/test")
    print(f"train: {len(train_set)}, valid: {len(valid_set)}, test: {len(test_set)}")

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(train_set, valid_set, test_set, batch_size=config["batch_size"], seed=config["seed"])

    # Initialize model and W&B
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(project="lambda_repformer", name=run_name)
    model = LambdaRepformer().to(device)
    wandb.watch(model, log_freq=100)

    images = {
        "bert_scene_narratives": torch.randn(config["batch_size"], 2, 768),
        "ada_scene_narratives": torch.randn(config["batch_size"], 2, 1536),
        "clip_images": torch.randn(config["batch_size"], 2, 512),
        "clip2d_images": torch.randn(config["batch_size"], 2, 1024, 14, 14),
        "vit_images": torch.randn(config["batch_size"], 2, 768),
        "dinov2_images": torch.randn(config["batch_size"], 2, 1024)
    }
    texts = {
        "bert": torch.randn(config["batch_size"], 768),
        "clip": torch.randn(config["batch_size"], 512),
        "ada": torch.randn(config["batch_size"], 1536)
    }
    summary(model, input_data=(images, texts))

    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-1) # weight_decay=1e-5
    criterion = nn.CrossEntropyLoss(weight=calculate_weights(train_set, device))

    del train_set, valid_set, test_set

    # Execute training, validation, and testing
    best_checkpoint_path = run_training(model, train_loader, valid_loader, test_loader, optimizer, criterion, device, config)
    print(f"Best checkpoint: {best_checkpoint_path}")
    test_model_accuracy(model, config['dataset_name'], test_loader, best_checkpoint_path, device)

    wandb.finish()

def calculate_weights(dataset, device):
    # Calculate class weights for imbalance handling
    label_distribution = dataset.get_label_ratio()
    total_samples = sum(label_distribution.values())
    weights = torch.tensor([total_samples / label_distribution[cls] for cls in ["False", "True"]], device=device)
    return weights

def run_training(model, train_loader, valid_loader, test_loader, optimizer, criterion, device, config):
    # Training and validation loop with early stopping
    best_acc = 0
    counter = 0
    checkpoint_dir = create_checkpoint_dir()
    best_checkpoint_path = ""

    for epoch in range(config["max_epoch"]):
        train_model(model, train_loader, optimizer, criterion, device, epoch)
        valid_acc, _ = validate_model(model, valid_loader, criterion, device, epoch)
        print(f"Validation Accuracy: {valid_acc}")
        # train_acc, _ = validate_model(model, train_loader, criterion, device, epoch, split="train")
        test_acc, _ = validate_model(model, test_loader, criterion, device, epoch, split="test")
        print(f"Test Accuracy: {test_acc}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_model.pth")
            save_checkpoint(model, best_checkpoint_path)
            counter = 0
        else:
            counter += 1

    return best_checkpoint_path

def test_model_accuracy(model, dataset_name, test_loader, checkpoint_path, device):
    # Load best model and evaluate on test set
    # TODO: Add confusion matrix and per-task accuracy
    # TODO: Add text file for qualitative results
    load_checkpoint(model, checkpoint_path)
    test_acc = test_model(model, test_loader, device, dataset_name=dataset_name)
    print(f"Test Accuracy: {test_acc}")

if __name__ == "__main__":
    main()
