import numpy as np
import cv2
import matplotlib.pyplot as plt
#from skimage import io

# 안개 낀 사진의 depth map을 구하기 위해,
# 사진의 transmission rate을 구해서 depth를 구한다.

# 해당 코드는 "Single Image Haze Removal Using Dark Channel Prior"의 논문 내용을 참고로 했다.



def get_dark_channel(image, mask_size):

    # Image의 Dark Channel Prior를 구한다.

    row, col, dump = image.shape
    padding = np.pad(image, ((int(mask_size / 2), int(mask_size / 2)), (int(mask_size / 2), int(mask_size / 2)), (0, 0)), 'edge')
    dark_chan = np.zeros((row,col))
    for i,j in np.ndindex(dark_chan.shape):
        dark_chan[i, j] = np.min(padding[i:i + mask_size, j:j + mask_size, :])

    # Return : M * N 배열, dark channel prior ([0, L-1]).
    return dark_chan


def get_atmosphere(image, dark_chan, pixels):

    # Image의 Atmosphere light (A값)을 구한다.

    row, col = dark_chan.shape
    flat_image = image.reshape(row * col, 3)
    flat_dark = dark_chan.ravel()
    searchidx = (-flat_dark).argsort()[:row * col]

    # Return : 각 channel의 A값을 가지는 배열
    return np.max(flat_image.take(searchidx, axis = 0), axis = 0)

def get_transmission(image, atmosphere, omega, mask_size):

    # Image의 Transmission Rate을 구한다.

    return 1 - omega * get_dark_channel(image / atmosphere, mask_size)

def get_depth(transmission, scatter_coefficient):

    # Transmission Rate를 바탕으로 Depth를 구한다.

    return (-scatter_coefficient) * np.log(transmission)

def drawFigure(loc, img, label):
    plt.subplot(*loc), plt.imshow(img, cmap='gray')
    plt.title(label), plt.xticks([]), plt.yticks([])

def depth_hazy_clear(hazy_image, clear_image, a, beta):
    dividend_r = -(np.log(hazy_image[:,:,0] - a) - np.log(clear_image[:,:,0] - a)) / beta
    dividend_g = -(np.log(hazy_image[:,:,1] - a) - np.log(clear_image[:,:,1] - a)) / beta
    dividend_b = -(np.log(hazy_image[:,:,2] - a) - np.log(clear_image[:,:,2] - a)) / beta
    depth = dividend_b + dividend_g + dividend_r / 3
    plt.imshow(depth, cmap="gray")
    plt.show()

# 사진을 받아서, 대략적 depth 분포를 볼 것이다.
H = cv2.imread('HEB_Bing_091_fake_A.png')
D = cv2.imread('HEB_Bing_091_real_B.png')
mask_size = 5
# 사진의 dark channel prior를 구한다. 논문에서는 patch size를 15x15로 실험했다.
image_dark_channel = get_dark_channel(H, mask_size)

# A를 구할 때, 상위 0.1% 안에 드는 픽셀들을 고른다.
atmosphere_light = get_atmosphere(H, image_dark_channel, 0.01)

# Transmission rate (T(x))를 구한다. 논문에서는 omega 값을 0.95로 설정했다.
transmission_rate = get_transmission(H, atmosphere_light, 0.95, mask_size)

# d(x)를 구한다. 산란 계수를 정해야 한다.
'''
drawFigure((2, 3, 1), get_depth(transmission_rate, 0.4), "0.4")
drawFigure((2, 3, 2), get_depth(transmission_rate, 0.6), "0.6")
drawFigure((2, 3, 3), get_depth(transmission_rate, 0.8), "0.8")
drawFigure((2, 3, 4), get_depth(transmission_rate, 1.0), "1.0")
drawFigure((2, 3, 5), get_depth(transmission_rate, 1.2), "1.2")
drawFigure((2, 3, 6), get_depth(transmission_rate, 1.4), "1.4")
'''
#depth = depth_hazy_clear(H, D, 1.0, 1.6)
depth = get_depth(transmission_rate, 0.8)
plt.imshow(depth, cmap="gray")
plt.show()
'''
cv2.imshow('test', depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
