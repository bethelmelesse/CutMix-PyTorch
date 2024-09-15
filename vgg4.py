#Fashion MNIST with Pytorch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import sklearn.metrics as metrics
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from skimage.transform import pyramid_reduce, pyramid_expand, resize, pyramid_gaussian
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_csv = pd.read_csv("data/fashion-mnist_train.csv")
test_csv = pd.read_csv("data/fashion-mnist_test.csv")

# LAPLACIAN PYRAMID FUNCTIONS
def get_gaussian_pyramid(image, max_level):
    return list(pyramid_gaussian(image, max_layer=max_level, downscale=2))

def get_laplacian_pyramid(image, max_level):
    gaussian_pyramid = get_gaussian_pyramid(image, max_level)
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(max_level, 0, -1):
        L = gaussian_pyramid[i - 1] - resize(pyramid_expand(gaussian_pyramid[i]), gaussian_pyramid[i - 1].shape)
        laplacian_pyramid.append(L)
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

# FASHION DATASET CLASS
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

# DATA AUGMENTATION AND DATALOADER
transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(28, padding=3),
    transforms.ToTensor(),
])

train_set = FashionDataset(train_csv, transform=transform, blend=True)
test_set = FashionDataset(test_csv, transform=transform)
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# CNN MODEL DEFINITION
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# MODEL INITIALIZATION
model = FashionCNN()
model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# TRAINING THE MODEL
num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        outputs = model(train)
        loss = error(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        count += 1
    
        if not (count % 50):
            total = 0
            correct = 0
        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
            
                test = Variable(images.view(100, 1, 28, 28))
                outputs = model(test)
            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
            
                total += len(labels)
            
            accuracy = (correct.cpu().numpy() * 100) / total

            loss_list.append(loss.data.cpu().numpy())  # Modification here
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        
        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))


loss_cpu = [l for l in loss_list]  
plt.plot(iteration_list, loss_cpu)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Iterations vs Loss")
plt.show()



plt.plot(iteration_list, accuracy_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy")
plt.show()


# # PLOTTING LOSS AND ACCURACY
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(loss_list, label='Loss')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(accuracy_list, label='Accuracy')
# plt.title('Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# VISUALIZING SOME TEST IMAGES AND THEIR PREDICTIONS
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)

# Display images and labels
fig, axes = plt.subplots(figsize=(28, 28), ncols=4, nrows=4)
for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
    # ax.title.set_text('Predicted: {}'.format(str(torch.argmax(outputs[i]))))
plt.show()

# EVALUATION ON TEST DATA
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# CONFUSION MATRIX AND CLASSIFICATION REPORT
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