import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.transform import pyramid_reduce, pyramid_laplacian, pyramid_expand, resize
import torch.nn.functional as F


def get_gaussian_pyramid(image):
    rows, cols, dim = image.shape
    gaussian_pyramid = [image]
    while rows > 1 and cols > 1:
        #print(rows, cols)
        image = pyramid_reduce(image, downscale=2, channel_axis=-1)
        gaussian_pyramid.append(image)
        #print(image.shape)
        rows //= 2
        cols //= 2
    return gaussian_pyramid


def get_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = [gaussian_pyramid[len(gaussian_pyramid)-1]]
    for i in range(len(gaussian_pyramid)-2, -1, -1):
        image = gaussian_pyramid[i] - resize(pyramid_expand(gaussian_pyramid[i+1], channel_axis=-1), gaussian_pyramid[i].shape)
        #print(i, image.shape)
        laplacian_pyramid.append(np.copy(image))
    laplacian_pyramid = laplacian_pyramid[::-1]
    return laplacian_pyramid

def reconstruct_image_from_laplacian_pyramid(pyramid):
    i = len(pyramid) - 2
    prev = pyramid[i+1]
    #plt.figure(figsize=(20,20))
    j = 1
    while i >= 0:
        prev = resize(pyramid_expand(prev, upscale=2, channel_axis=-1), pyramid[i].shape)
        # prev = resize(pyramid_expand(prev, upscale=2), pyramid[i].shape)
        im = np.clip(pyramid[i] + prev,0,1)
        #plt.subplot(3,3,j)
        # plt.imshow(im)
        # plt.title('Level=' + str(j) + ', ' + str(im.shape[0]) + 'x' + str(im.shape[1]), size=20)
        prev = im
        i -= 1
        j += 1
    return im

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    
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


# load the images in this folder using dataloader
A = imread('images/apple.png')[...,:3] / 255
B = imread('images/orange.png')[...,:3] / 255


# apply CutMix on these images
lam = np.random.beta(1, 1)
print(lam)

# rand_index = torch.randperm(input.size()[0]).cuda() # generate a random index，

# transfer the data to tensor
# img1 = transforms.ToTensor()(A)
# img2 = transforms.ToTensor()(B)
# target_a = img1
# target_b = img2

size = A.shape
bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)

# CutMix the images
# target_a[:, bby1:bby2, bbx1:bbx2] = target_b[:, bby1:bby2, bbx1:bbx2]

# creating Mask image
white_image = np.ones_like(A)
# rectangle_position = [(50, 50), (250, 250)]
rectangle_position = [(bbx1, bby1), (bbx2, bby2)]
white_image[rectangle_position[0][1]:rectangle_position[1][1], rectangle_position[0][0]:rectangle_position[1][0]] = 0
M = white_image

# transfer back to PIL image
# target_a = transforms.ToPILImage()(target_a)


# show the result
# plt.imshow(target_a)
# plt.show()

rows, cols, dim = A.shape
pyramidA = get_laplacian_pyramid(get_gaussian_pyramid(A))
pyramidB = get_laplacian_pyramid(get_gaussian_pyramid(B))
pyramidM = get_gaussian_pyramid(M)

pyramidC = []
for i in range(len(pyramidM)):
    
    im = pyramidM[i]*pyramidA[i] + (1-pyramidM[i])*pyramidB[i]

    #print(np.max(im), np.min(im), np.mean(im))
    pyramidC.append(im)

im = reconstruct_image_from_laplacian_pyramid(pyramidC)


f, axarr = plt.subplots(2,2)

axarr[0,0].imshow(A)
axarr[0,1].imshow(B)
axarr[1,0].imshow(M)
axarr[1,1].imshow(im)

# plt.axis('off')
# plt.imshow(M)
plt.show()
print()
