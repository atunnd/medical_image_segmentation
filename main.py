import argparse
import torch
from model.unet3d import UNet3D
from model.vnet import VNet
import math
import torch
from torch.nn import CrossEntropyLoss, Module
from load_data import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transforms import train_transform, val_transform
import os
import shutil

# Allow the program to continue despite OpenMP runtime conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class DiceLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def main():
    parser = argparse.ArgumentParser(description="Train a 3D segmentation model")
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'vnet'], help='Model type to use')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--training_batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--testing_batch_size', type=int, default=16, help='Testing batch size')
    parser.add_argument('--train_cuda', type=bool, default=True, help='Use CUDA for training')
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'dice'], help='Loss function to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--bce_training_weight', type=float, default=0.996, help='Weight for BCE loss')
    parser.add_argument('--num_worker', type=int, default=0)

    args = parser.parse_args()

    # Set device based on the argument
    device = 'cuda' if args.train_cuda and torch.cuda.is_available() else 'cpu'

    # Select model
    if args.model == 'unet':
        model = UNet3D(1, 2).to(device)
    elif args.model == 'vnet':
        model = VNet().to(device)
    else:
        raise ValueError(f"Model {args.model} is not supported")

    # Set up TensorBoard
    if os.path.exists("runs"):
        shutil.rmtree("runs")
    writer = SummaryWriter("runs")
    os.makedirs(f'checkpoints/{args.model}', exist_ok=True)

    # Load data
    train_dataloader, val_dataloader = get_train_val_test_Dataloaders(
        args.data_path, args.training_batch_size, args.testing_batch_size, args.num_worker
    )

    # Initialize optimizer and loss function
    optimizer = Adam(params=model.parameters())
    if args.loss == 'bce':
        bce_weights = torch.Tensor([1 - args.bce_training_weight, args.bce_training_weight]).to(device)
        criterion = CrossEntropyLoss(weight=bce_weights)
    elif args.loss == 'dice':
        criterion = DiceLoss()

    min_valid_loss = math.inf
    stop_count = 0

    print("Starting training")

    # Training loop
    for epoch in range(args.epoch):
        print(f"Epoch {epoch + 1}")
        train_loss = 0.0
        model.train()

        for i, data in enumerate(train_dataloader):
            images, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"Training batch {i} loss: {loss.item()}")

        valid_loss = 0.0
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                images, labels = data['image'].to(device), data['label'].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                print(f"Validation batch {i} loss: {loss.item()}")

        avg_train_loss = train_loss / len(train_dataloader)
        avg_valid_loss = valid_loss / len(val_dataloader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_valid_loss, epoch)

        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss} Validation Loss: {avg_valid_loss}")

        # Save model if validation loss improves
        if min_valid_loss > avg_valid_loss:
            stop_count = 0
            print(f"Validation Loss Decreased({min_valid_loss:.6f} ---> {avg_valid_loss:.6f}), saving model.")
            min_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), f'checkpoints/{args.model}/epoch{epoch + 1}_valLoss{min_valid_loss:.6f}.pth')
        else:
            stop_count += 1
            if stop_count == 5:
                print("Early stopping triggered")
                break

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
