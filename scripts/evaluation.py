from torchmetrics import Accuracy
from tqdm import tqdm
import torch


def evaluation(model, val_loader, loss_fn):
    """
    Evaluates a classification model measuring its accuracy and average loss
    using the provided loss function.

    :param model: The model to evaluate
    :param val_loader: DataLoader for the validation dataset
    :param loss_fn: Loss function to use for evaluation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    num_classes = model.num_classes

    # Initialize metrics
    accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, colour='red'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)

            # Update metrics
            accuracy.update(preds, labels)

    # Compute final metric results
    final_accuracy = accuracy.compute()

    # Average loss over all validation data
    average_loss = total_loss / len(val_loader.dataset)

    return {
        'accuracy': final_accuracy.item(),
        'average_loss': average_loss
    }
