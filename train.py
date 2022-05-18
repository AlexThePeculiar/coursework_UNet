import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from torchvision import transforms
import utils

# Global parameters
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def train_fn(loaded_data, model, optimizer, loss_fn, scaler):
    data_loop = tqdm(loaded_data)

    for batch_id, (data, targets) in enumerate(data_loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm
        data_loop.set_postfix(loss=loss.item())


def train():
    edit = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = utils.get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE,
        train_transform=edit, val_transform=edit,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    if LOAD_MODEL:
        utils.load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }

        utils.save_checkpoint(checkpoint)
        utils.check_accuracy(val_loader, model, device=DEVICE)

if __name__ == "__main__":
    train()

