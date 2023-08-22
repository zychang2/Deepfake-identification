# Libraries
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed may be slowed down")
    device = 'cpu'

class BinaryClassificationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label



# print("Setting up model:")

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 128x128x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 64x64x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 32x32x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 16x16x256
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 8x8x256
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 4x4x512
        
        # Below 2 layers are used if image size is (1024, 1024, 3)
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 8x8x512
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # 4x4x512
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(4 * 4 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = x.to(device)
        x = self.conv1(x)
        x = checkpoint(self.conv2, x)
        x = checkpoint(self.conv3, x)
        x = checkpoint(self.conv4, x)
        x = checkpoint(self.conv5, x)
        x = checkpoint(self.conv6, x)
        x = checkpoint(self.conv7, x)
        x = checkpoint(self.conv8, x)
        x = self.fc(x)
        return x


model = Network().to(device)

epochs = 10
loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-3
weight_decay = 1e-5 
optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)

def make_csv():
    # Preprocessing

    # Iterate through fake images
    
    fake_img_dir = f"{os.environ['DATA_DIR']}dct_fake_imgs/"
    real_img_dir = f"{os.environ['DATA_DIR']}dct_real_imgs/"

    data = [['path', 'class']]

    for filename in os.listdir(fake_img_dir):
        path = f"{fake_img_dir}{filename}"
        data.append([path, 0])

    for filename in os.listdir(real_img_dir):
        path = f"{real_img_dir}{filename}"
        data.append([path, 1])

    with open(os.environ['CSV_DIR'], 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(data)


def compute_mean_std(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)

    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    return mean, std

def make_data_loaders(Mean, Std):
    data = pd.read_csv(os.environ['CSV_DIR'], header=0, names=['path', 'class'])

    # Separate the data into two classes
    class_0 = data[data['class'] == 0]
    class_1 = data[data['class'] == 1]

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    # Split class_0
    class_0_train, class_0_val_test = train_test_split(class_0, test_size=(val_ratio + test_ratio))
    class_0_val, class_0_test = train_test_split(class_0_val_test, test_size=test_ratio/(val_ratio + test_ratio))

    # Split class_1
    class_1_train, class_1_val_test = train_test_split(class_1, test_size=(val_ratio + test_ratio))
    class_1_val, class_1_test = train_test_split(class_1_val_test, test_size=test_ratio/(val_ratio + test_ratio))

    train_data = pd.concat([class_0_train, class_1_train]).sample(frac=1).reset_index(drop=True)
    val_data = pd.concat([class_0_val, class_1_val]).sample(frac=1).reset_index(drop=True)
    test_data = pd.concat([class_0_test, class_1_test]).sample(frac=1).reset_index(drop=True)

    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(Mean / 255.0, Std / 255.0) 
                    ])

    # Create the Datasets
    train_dataset = BinaryClassificationDataset(train_data, transform=transform)
    val_dataset = BinaryClassificationDataset(val_data, transform=transform)
    test_dataset = BinaryClassificationDataset(test_data, transform=transform)

    batch_size = 16

    train_dl = DataLoader(train_dataset, batch_size, shuffle = True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_dl, val_dl, test_dl


def train(model, trainloader, valloader, epochs):
  print("Start training...")
  trn_loss_hist = []
  trn_acc_hist = []
  val_acc_hist = []
  for i in range(epochs):
      running_loss = []
      print('-----------------Epoch = %d-----------------' % (i+1))
      for batch, label in tqdm(trainloader):
          batch = batch.to(device)
          label = label.to(device)
          optimizer.zero_grad()  # Clear gradients from the previous iteration
          # This will call Network.forward() that you implement
          pred = model(batch)
          loss = loss_function(pred, label)  # Calculate the loss
          running_loss.append(loss.item())
          loss.backward()  # Backprop gradients to all tensors in the network
          optimizer.step()  # Update trainable weights
      print("\n Epoch {} loss:{}".format(i+1, np.mean(running_loss)))

      # Keep track of training loss, accuracy, and validation loss
      trn_loss_hist.append(np.mean(running_loss))
      trn_acc_hist.append(evaluate(model, trainloader))
      print("\n Evaluate on validation set...")
      val_acc_hist.append(evaluate(model, valloader))
      
  print("Done!")
  return trn_loss_hist, trn_acc_hist, val_acc_hist


def evaluate(model, loader):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
        acc = correct/len(loader.dataset)
        print("\n Evaluation accuracy: {}".format(acc))
        return acc

def main():
    make_csv()
    image_dirR = f"{os.environ['DATA_DIR']}dct_real_imgs" 
    image_dirF = f"{os.environ['DATA_DIR']}dct_fake_imgs"
    num_images = 0
    mean = np.zeros(1)
    std = np.zeros(1)
    for filename in os.listdir(image_dirR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dirR, filename)
            image_mean, image_std = compute_mean_std(image_path)
            mean += image_mean
            std += image_std
            num_images += 1

    for filename in os.listdir(image_dirF):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dirF, filename)
            image_mean, image_std = compute_mean_std(image_path)
            mean += image_mean
            std += image_std
            num_images += 1

    Mean = mean / num_images
    Std = std / num_images
    print("Mean of images: ", Mean)
    print("Std of images: ", Std)

    train_dl, val_dl, test_dl = make_data_loaders(Mean, Std)
    trn_loss_hist, trn_acc_hist, val_acc_hist = train(model, train_dl,
                                                  val_dl, epochs)

    print("\n Evaluate on test set")
    evaluate(model, test_dl)




if __name__ == "__main__":
    main()