import torch
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import time
import datetime
import os

from src.model import ContrastiveLambdaRepformer
from utils.data_loader import CustomDataset
from utils.utils import torch_fix_seed, load_checkpoint, plot_confusion_matrices

def test_model(model, test_loader, device, dataset_name="SP-RT-1"):
    task_metrics = { 'correct': {}, 'total': {}, 'TP': {}, 'FP': {}, 'FN': {} }
    task_name = dataset_name
    os.makedirs("res", exist_ok=True)
    resfile = f"res/{dataset_name}_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(resfile, 'a') as f:
        f.write(f"file_name, predicted, gt, confusion\n")
    with torch.no_grad():
        for images, texts, image_paths, target in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            output = model(images, texts).to(device)
            target = target.to(device)
            _, predicted = torch.max(output, 1)
            if predicted.nelement() == 0 or target.nelement() == 0:  # More efficient check for empty tensors
                continue

            update_task_metrics(predicted, target, image_paths, task_metrics, dataset_name, resfile)

    print_task_metrics(task_metrics, dataset_name)
    return calculate_test_accuracy(task_metrics)

def update_task_metrics(predicted, target, image_paths, task_metrics, dataset_name, resfile=None):
    for i, image_path in enumerate(image_paths[0]):
        task_name = derive_task_name(image_path, dataset_name)
        metrics = task_metrics

        metrics['total'].setdefault(task_name, 0)
        metrics['correct'].setdefault(task_name, 0)
        metrics['TP'].setdefault(task_name, 0)
        metrics['FP'].setdefault(task_name, 0)
        metrics['FN'].setdefault(task_name, 0)

        metrics['total'][task_name] += 1
        correct_prediction = predicted[i] == target[i]
        if correct_prediction:
            metrics['correct'][task_name] += 1
        if (predicted[i] == 1) & (target[i] == 1):
            confusion = "TP"
        elif (predicted[i] == 1) & (target[i] == 0):
            confusion = "FP"
        elif (predicted[i] == 0) & (target[i] == 1):
            confusion = "FN"
        else:
            confusion = "TN"
        if resfile:
            with open(resfile, 'a') as f:
                f.write(f"{image_path}, {predicted[i].item()}, {target[i].item()}, {confusion}\n")
        metrics['TP'][task_name] += ((predicted[i] == 1) & (target[i] == 1)).item()
        metrics['FP'][task_name] += ((predicted[i] == 1) & (target[i] == 0)).item()
        metrics['FN'][task_name] += ((predicted[i] == 0) & (target[i] == 1)).item()

def derive_task_name(image_path, dataset_name):
    return dataset_name

def print_task_metrics(task_metrics, dataset_name):
    for task_name in task_metrics['correct']:
        accuracy = task_metrics['correct'][task_name] / task_metrics['total'][task_name] * 100  # Convert to percentage
        print(f"Task: {task_name}, Correct: {task_metrics['correct'][task_name]}/{task_metrics['total'][task_name]}, Accuracy: {accuracy:.2f}%")
    plot_confusion_matrices(list(task_metrics['correct'].keys()),
                            {name: (task_metrics['TP'][name], task_metrics['FP'][name], task_metrics['FN'][name], task_metrics['total'][name]) for name in task_metrics['correct']},
                            f'{dataset_name}_confusion_matrix.png')

def calculate_test_accuracy(task_metrics):
    total = sum(task_metrics['total'].values())
    correct = sum(task_metrics['correct'].values())
    return correct / total if total > 0 else 0

def main():
    with open("configs/config.json") as config_file:
        config = json.load(config_file)
    if config["seed"] != False:
        torch_fix_seed(config["seed"])
    # torch_fix_seed(config["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveLambdaRepformer().to(device)

    if config["dataset_name"] == "SP-RT-1":
        test_set = CustomDataset(f"{config['data_path']}/{config['dataset_name']}/test")
    elif config["dataset_name"] == "SP-HSR":
        test_set = CustomDataset(f"{config['data_path']}/{config['dataset_name']}/hsr_data")
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    checkpoint_path = config["checkpoint_path"]
    load_checkpoint(model, checkpoint_path)
    start_time = time.time()
    test_acc = test_model(model, test_loader, device, dataset_name=config["dataset_name"])
    end_time = time.time()
    print(f"Time: {end_time - start_time:.2f} seconds")
    print(f"Overall Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
