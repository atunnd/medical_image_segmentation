import argparse
import torch
from model.unet3d import UNet3D
from model.vnet import VNet
import math
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from load_data import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model.unet3d import UNet3D
from model.vnet import VNet
from transforms import (train_transform, val_transform)
import os
import shutil

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--model', type=str, default=['unet'], choices=['unet', 'vnet'], help='Model type to use')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--training_batch_size', type=str, default=16, help="Training batch size")
    parser.add_argument('--testing_batch_size', type=str, default=16, help="Testing batch size")
    parser.add_argument('--train_cuda', type=bool, default=True, help="Training on cuda")
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'dice'], help='loss function')
    parser.add_argument('--data_path', type=str, help='Path to directory for training and testing data')
    parser.add_argument('--bce_training_weight', type=str, default=0.996, help='Setting training weight for bce')
    parser.add_argument()
    # Parse the arguments
    args = parser.parse_args()

    if args.train_cuda == True:
        device = 'cuda'
    else:
        device='cpu'

    if args.model == 'unet':
        model = UNet3D()
        model = model.to(device)
    elif args.model == 'vnet':
        model = VNet()
        model = model.to(device)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    if os.path.exists("runs"):
        shutil.rmtree("runs")
    writer = SummaryWriter("runs")
    os.makedirs(f'checkpoints/{args.model}', exist_ok=True)

    train_dataloader, val_dataloader = get_train_val_test_Dataloaders(
                                        args.data_path,
                                        args.training_batch_size,
                                        args.testing_batch_size)
    optimizer = Adam(params=model.parameters())
    if args.loss == 'bce':
        bce_weights = [1 - args.bce_training_weight, args.bce_training_weight]
        criterion = CrossEntropyLoss(weight=torch.Tensor(bce_weights).to(device))
    elif args.loss == 'dice':
        criterion = DiceLoss()

    min_valid_loss = math.inf
    stop_count = 0

    print(f"Initialize training")

    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        train_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader):
            image, ground_truth = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            target = model(image)
            loss = criterion(target, ground_truth)
            loss.backward()
            optimizer.step()
            
            print(f"Training batch {i} has loss: {loss}")

            train_loss += loss.item()

        valid_loss = 0.0
        model.eval()
        for i, data in enumerate(val_dataloader):
            image, ground_truth = data['image'].to(device), data['label'].to(device)
            target = model(image)
            loss = criterion(target, ground_truth)
            print(f"Validate batch {i} has loss: {loss}")
            valid_loss += loss.item()

        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_loss /
                        len(val_dataloader), epoch)

        print(f'=> General {epoch} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')

        if min_valid_loss > valid_loss:
            stop_count = 0
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), f'checkpoints/{args.model}/epoch{epoch}_valLoss{min_valid_loss}.pth')
        else:    
            stop_count += 1
            if stop_count == 5:
                print("Early stopping triggered")
                break
        print("\n")
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
