# Implement CNN for edge detection
# train on BSD500 dataset

#compare classical methods vs DL
#visualize performance metrics(F1, Precision, Recall)

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
#compare my canny edge detector to open cv to CNN based on a gray scale image

## Canny Edge Steps
# Blur image with gaussian filter -> to remove noise
# Calculate X and Y partial Derivatives with a filter
# Calculate Gradient magnitude and orientation with Gx and Gy
# Non maximum supression 
# Double thresholding (Hysteresis)
# Keep weak edges that are connected to strong edges. 

#followed a medium tutorial to learn how to implement canny edge with pytorch
#uncescessary, but thought it would be an interesting way to learn this again

def get_gaussian_kernel(k=3,  mu=0, sigma=1, normalize=True):
    """
    Increasing kernel size increases blur of output
    """
    # k = size of the output matrix
    # sigma = float of sigma to calculate kernel
    gaussian_1D  = np.linspace(-1,1,k)
    #compute a grid distsnce from center 
    x,y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x**2 + y**2) **.5 # this distance formula was not in original implentation
    #compute 2d gaussian
    gaussian_2D_numerator = np.exp(-(distance-mu)**2/(2*sigma**2))
    gaussian_2D = gaussian_2D_numerator/(2*np.pi*sigma**2)
    #normalizing is done so that sum of all elemenrts is one
    # this means convolving won't change overall intensity. 
    if normalize:
        gaussian_2D /= np.sum(gaussian_2D)

    return gaussian_2D

def get_sobel_kernel(k=3):
    #gives the horizontal filter| brighter pixels on right compared with left
    #get verticle with transpose | brighter pixels on top compared with bottom

    # get range
    range = np.linspace(-(k//2), k//2, k)
    #compute a grid the numerator and the axis-distances
    x,y = np.meshgrid(range, range) # this is a weird function it makes sense but its odd. 
    denominator = (x**2 + y **2)
    denominator[:,k//2] = 1 # prevent division by 0
    sobel_2D_X = x / denominator
    sobel_2D_Y = sobel_2D_X.T
    return sobel_2D_X, sobel_2D_Y

def get_thin_kernels(start=0, end=360, step=45):
        """
        gets neighborhood kernels for the thin edges
        start: starting angle in degrees
        """

        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels









def main():
    # load image 
    image = cv2.imread("El_Castillo.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Could not load image. Using a test pattern.")
        image = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    else:
        image = cv2.resize(image, (500,500))
    
    og_canny = cv2.Canny(image, 50, 150)

    # display image and destroy on click
    cv2.imshow("OpenCV Canny", og_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0
if __name__ == "__main__":
    main()