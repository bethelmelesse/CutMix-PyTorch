import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from skimage.transform import pyramid_reduce, pyramid_expand, resize
import random
from torchvision.transforms import ToTensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
train_csv = pd.read_csv('C:\\Users\\user\\Desktop\\Ajou Files\\Semesters\\Fall 2023\\Advanced Machine Learning\\Project Files\\Fashion MNIST Practice\\fashion-mnist_train.csv\\fashion-mnist_train.csv')
test_csv = pd.read_csv('C:\\Users\\user\\Desktop\\Ajou Files\\Semesters\\Fall 2023\\Advanced Machine Learning\\Project Files\\Fashion MNIST Practice\\fashion-mnist_test.csv\\fashion-mnist_test.csv')

def mnist():
    training_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
        blend=True
    )

    test_data = datasets.FashionMNIST(
        root='../data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    num_classes = 10
    return training_data, test_data, num_classes

# Gaussian and Laplacian Pyramid Functions
def get_gaussian_pyramid(image, max_level):
    gaussian_pyramid = [image]
    for _ in range(max_level):
        image = pyramid_reduce(image, downscale=2)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def get_laplacian_pyramid(image, max_level):
    gaussian_pyramid = get_gaussian_pyramid(image, max_level)
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(max_level, 0, -1):
        expanded = resize(pyramid_expand(gaussian_pyramid[i]), gaussian_pyramid[i - 1].shape)
        laplacian = gaussian_pyramid[i - 1] - expanded
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid[::-1]

def blend_images(image1, image2, mask, max_level=5):
    pyramid1 = get_laplacian_pyramid(image1, max_level)
    pyramid2 = get_laplacian_pyramid(image2, max_level)
    mask_pyramid = get_gaussian_pyramid(mask, max_level)

    blended_pyramid = []
    for l1, l2, m in zip(pyramid1, pyramid2, mask_pyramid):
        blended = l1 * m + l2 * (1 - m)
        blended_pyramid.append(blended)

    blended_image = blended_pyramid[-1]
    for i in range(max_level - 1, -1, -1):
        blended_image = resize(pyramid_expand(blended_image), blended_pyramid[i].shape) + blended_pyramid[i]

    return blended_image

# FashionDataset Class
class FashionDataset(Dataset):
    def __init__(self, data, transform=None, blend=False):
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        self.blend = blend
        label = []
        image = []
        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.blend:
            index2 = random.randint(0, len(self.images) - 1)
            image2 = self.images[index2]
            mask = np.random.rand(28, 28, 1)
            image = image.astype(np.float32)  # Convert to float32

        if self.transform is not None:
            image = self.transform(image)

        return image, label

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

# Data Augmentation
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(28, padding=3),
    transforms.ToTensor(),
])

# Dataset and DataLoader

train_set, test_set, num_classes = mnist()

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# CNN Model Definition
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# Instantiate the model, error function, and optimizer
model = FashionCNN().to(device)
error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the Model
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = error(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy of the network on test images: {accuracy * 100}%')

# Confusion Matrix and Classification Report
all_labels = []
all_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print(confusion_matrix(all_labels, all_predictions))
print(classification_report(all_labels, all_predictions))