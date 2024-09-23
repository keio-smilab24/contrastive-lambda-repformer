import torch
from tqdm import tqdm
import wandb

def train_model(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    num_zeros, num_ones = 0, 0
    true_predicted, false_predicted = 0, 0
    correct = 0

    for batch_idx, (images, texts, image_paths, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        output = model(images, texts)
        _, predicted = torch.max(output, 1)
        true_predicted += (predicted == 1).sum().item()
        false_predicted += (predicted == 0).sum().item()
        loss = criterion(output.to(device), target.to(device))
        loss.backward()
        optimizer.step()


        if (epoch == 0 and batch_idx >= 100) or epoch > 0:
            correct += (predicted.to(device) == target.to(device)).sum().item()
            train_acc = correct / (true_predicted + false_predicted) if (true_predicted + false_predicted) > 0 else 0

            wandb.log({"train_loss": loss.item(), "train_acc": train_acc})

        num_zeros += (target == 0).sum().item()
        num_ones += (target == 1).sum().item()

    # print(f"--train-- True predicted: {true_predicted}, False predicted: {false_predicted}")
