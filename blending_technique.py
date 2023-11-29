# % matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import pyramid_reduce, pyramid_laplacian, pyramid_expand, resize
import numpy as np
import matplotlib.pyplot as plt


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
        image = gaussian_pyramid[i] - resize(pyramid_expand(gaussian_pyramid[i+1]), gaussian_pyramid[i].shape)
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
        # prev = resize(pyramid_expand(prev, upscale=2, channel_axis=-1), pyramid[i].shape)
        prev = resize(pyramid_expand(prev, upscale=2), pyramid[i].shape)
        im = np.clip(pyramid[i] + prev,0,1)
        #plt.subplot(3,3,j)
        # plt.imshow(im)
        # plt.title('Level=' + str(j) + ', ' + str(im.shape[0]) + 'x' + str(im.shape[1]), size=20)
        prev = im
        i -= 1
        j += 1
    return im



A = imread('images/apple.png')[...,:3] / 255
B = imread('images/orange.png')[...,:3] / 255
M = imread('images/mask.png')[...,:3] / 255

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

plt.axis('off')
plt.show()