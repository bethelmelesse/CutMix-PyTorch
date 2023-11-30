import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# load the images in this folder using dataloader
img1 = Image.open('images/1.jpg')
img2 = Image.open('images/2.jpg')


def rand_bbox(size, lam):
    W = 512
    H = 512
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# apply CutMix on these images
lam = np.random.beta(1, 1)
print(lam)
# rand_index = torch.randperm(input.size()[0]).cuda() # generate a random index，
# transfer the data to tensor
img1 = transforms.ToTensor()(img1)
img2 = transforms.ToTensor()(img2)
target_a = img1
target_b = img2
bbx1, bby1, bbx2, bby2 = rand_bbox(512, lam)
# CutMix the images
target_a[:, bby1:bby2, bbx1:bbx2] = target_b[:, bby1:bby2, bbx1:bbx2]

# transfer back to PIL image
target_a = transforms.ToPILImage()(target_a)

# show the result
plt.imshow(target_a)
plt.show()
