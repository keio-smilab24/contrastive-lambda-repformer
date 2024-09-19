import torch
import wandb

def validate_model(model, valid_loader, criterion, device, epoch, split="valid"):
    model.eval()
    correct, total = 0, 0
    total_valid_loss, TP, FP, FN = 0, 0, 0, 0
    num_zeros, num_ones = 0, 0
    true_predicted, false_predicted = 0, 0

    with torch.no_grad():
        for batch_idx, (images, texts, image_paths, target) in enumerate(valid_loader):
            output = model(images, texts).to(device)
            _, predicted = torch.max(output, 1)
            predicted = predicted.to(device)
            target = target.to(device)
            true_predicted += (predicted == 1).sum().item()
            false_predicted += (predicted == 0).sum().item()
            num_ones += (target == 1).sum().item()
            num_zeros += (target == 0).sum().item()
            loss = criterion(output, target)
            total_valid_loss += loss.item()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            TP += ((predicted == 1) & (target == 1)).sum().item()
            FP += ((predicted == 1) & (target == 0)).sum().item()
            FN += ((predicted == 0) & (target == 1)).sum().item()

    avg_valid_loss = total_valid_loss / len(valid_loader)
    valid_acc = 100 * correct / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if split == "test":
        print(f"--test-- True predicted: {true_predicted}, False predicted: {false_predicted}")
        print(f"Avg Test Loss {avg_valid_loss}")
        wandb.log({"test_loss": avg_valid_loss, "test_acc": valid_acc})
    elif split == "train":
        wandb.log({"train_loss": avg_valid_loss, "train_acc": valid_acc})
    else:
        print(f"--valid-- True predicted: {true_predicted}, False predicted: {false_predicted}")
        # print(f'--valid-- Epoch {epoch}: Number of zeros: {num_zeros}, Number of ones: {num_ones}')
        print(f"Avg Validation Loss {avg_valid_loss}")
        # print(f"F1 Score: {F1}")
        wandb.log({"valid_loss": avg_valid_loss, "valid_acc": valid_acc})

    return valid_acc, avg_valid_loss
