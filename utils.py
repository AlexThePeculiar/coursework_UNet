import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from train import IMAGE_WIDTH, IMAGE_HEIGHT

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(" => saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print(" => loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dir, train_mask_dir, val_dir, val_mask_dir, batch_size,
                train_transform=None, val_transform=None, num_workers=None, pin_memory=None
                ):
    train_ds = CarvanaDataset(image_dir=train_dir,mask_dir=train_mask_dir,transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

    val_ds = CarvanaDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False,num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def check_accuracy(loader, model, threshold=0.5, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > threshold).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            accuracy += num_correct/num_pixels

    dice_score = dice_score/len(loader)
    accuracy = accuracy/len(loader)

    print(f"accuracy: {accuracy*100:.2f}")
    print(f"dice score: {dice_score*100:.2f}")

    model.train()


def predict_image(img_path, model, device="cuda"):
    model.eval()
    image = Image.open(img_path).convert("RGB")

    edit = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    ])

    x = edit(image).unsqueeze(0)

    with torch.no_grad():
        y = torch.sigmoid(model(x))

    model.train()
    return y

def save_prediction(img_path,  res_path, model, threshold=0.5, color=[255,0,0], device="cuda"):
    y = predict_image(img_path, model, device)
    alpha = (y>threshold).float().squeeze(0).numpy().copy().transpose(1,2,0)

    original_image = Image.open(img_path).convert("RGB")

    resizer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    ])

    image = resizer(original_image).numpy().copy().transpose(1,2,0)

    color_overlay = np.full(image.shape, color)

    res_image = color_overlay*alpha + image * 255 * (1-alpha)

    Image.fromarray(res_image.astype(np.uint8)).save(res_path)


