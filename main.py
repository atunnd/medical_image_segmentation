import math
import numpy as np
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from load_data import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)


writer = SummaryWriter("runs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
if torch.cuda.is_available() and TRAIN_CUDA:
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    print("Train on cuda")
elif not torch.cuda.is_available() and TRAIN_CUDA:
    train_transforms = train_transform
    val_transforms = val_transform
    print('cuda not available! Training initialized on cpu ...')



train_dataloader, val_dataloader = get_train_val_test_Dataloaders(
     train_transforms=train_transforms, val_transforms=val_transforms)

criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):

    train_loss = 0.0
    model.train()
    for data in train_dataloader:
        image, ground_truth = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        target = model(image)
        #print(np.shape(image), np.shape(ground_truth), np.shape(target))
        loss = criterion(target, ground_truth)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for data in val_dataloader:
        image, ground_truth = data['image'].to(device), data['label'].to(device)
        target = model(image)
        loss = criterion(target, ground_truth)
        print("\n")
        valid_loss = loss.item()

    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss /
                      len(val_dataloader), epoch)

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(),
                   f'/kaggle/working/checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()