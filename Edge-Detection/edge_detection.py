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
        better implementation of non maximum supression step
        basically gets the 8 values surrounding a number and puts it in a 3x3 matrix
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

class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        self.device = 'cuda' if use_cuda else 'cpu'

        # TODO: Create a Gaussian kernel and initialize a Conv2D for smoothing
        # Hint: Use get_gaussian_kernel to create a 2D kernel and apply it with a Conv2D layer.
        gaussian_kernel = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels =1,  # applies 2D convolution
                                         out_channels=1, 
                                         kernel_size = k_gaussian,  
                                         padding = k_gaussian//2,
                                         bias = False)
        self.gaussian_filter.weight.data[:] = torch.from_numpy(gaussian_kernel).float() #convolution kernel becomes input

        # TODO: Create Sobel filters for x and y gradients using get_sobel_kernel
        # Hint: One filter is used directly; the other is its transpose.
        soble_kernel_x, soble_kernel_y = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels =1,  # applies 2D convolution
                                         out_channels=1, 
                                         kernel_size = k_sobel,  
                                         padding = k_sobel//2,
                                         bias = False)
        self.sobel_filter_x.weight.data[:] = torch.from_numpy(soble_kernel_x).float()

        self.sobel_filter_y = nn.Conv2d(in_channels =1,  # applies 2D convolution
                                         out_channels=1, 
                                         kernel_size = k_sobel,  
                                         padding = k_sobel//2,
                                         bias = False)
        self.sobel_filter_y.weight.data[:] = torch.from_numpy(soble_kernel_y).float()

        # TODO: Create directional filters for non-maximum suppression
        # Hint: Use get_thin_kernels and stack the resulting 8 filters.
        directional_kernels = np.stack(get_thin_kernels())
        self.directional_filter = nn.Conv2d(in_channels =1,  # applies 2D convolution
                                         out_channels=8, 
                                         kernel_size = get_thin_kernels()[0].shape[0],  
                                         padding = get_thin_kernels()[0].shape[-1]//2,
                                         bias = False)
        self.directional_filter.weight.data[:, 0] = torch.from_numpy(directional_kernels).float()

        # TODO: Define a 3x3 kernel for hysteresis
        # Hint: A simple kernel like a box filter with added weight is enough.
        # originally I did this by categorizing a pixel as high 
        # if its higher than its 8 neigbors
        # this method suggests that we can use a final convolution filter to 
        # categorize pixel as high if product of convolution > 1 

        hysteresis = np.ones((3,3)) +.25
        self.hysteresis = nn.Conv2d(in_channels =1,  # applies 2D convolution
                                         out_channels=1, 
                                         kernel_size = hysteresis.shape[0],  
                                         padding = hysteresis.shape[0]//2,
                                         bias = False)
        self.hysteresis.weight.data[:] = torch.from_numpy(hysteresis).float() # make a tensor from numpy array

    
    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # Normalize input to [0,1] range
        img = img / 255.0
        
        B, C, H, W = img.shape
        # B = Batch size images in a batch
        # C = color channels
        # H = Height
        # W = Width

        # TODO: Allocate output tensors (blurred, grad_x, grad_y, etc.)
        # Hint: These should be zero tensors on the same device as the image.
        blurred = torch.zeros(B, C, H, W).to(self.device)
        grad_x = torch.zeros(B, 1, H, W).to(self.device)
        grad_y = torch.zeros(B, 1, H, W).to(self.device)
        grad_magnitude = torch.zeros(B, 1, H, W).to(self.device)
        grad_orientation = torch.zeros(B, 1, H, W).to(self.device)

        # Step 1: Gaussian smoothing
        for c in range(C):
            # TODO: Apply Gaussian blur to each channel
            # Hint: Use your Gaussian filter here.
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            # TODO: Accumulate gradients in x and y directions
            # Hint: Apply Sobel filters on the blurred image.
            grad_x += self.sobel_filter_x(blurred[:, c:c+1])
            grad_y += self.sobel_filter_y(blurred[:, c:c+1])

        # TODO: Compute average gradients across channels
        # Hint: Divide accumulated gradients by the number of channels
        grad_x /= C
        grad_y /= C

        # TODO: Compute gradient magnitude and orientation
        grad_magnitude = (grad_x **2 + grad_y**2) **0.5
        
        # Handle division by zero in atan2 and use atan2 instead of atan
        grad_orientation = torch.atan2(grad_y, grad_x)
        grad_orientation = grad_orientation * (180/np.pi) + 180 # convert to degree [0, 360]
        grad_orientation = torch.round(grad_orientation/45) * 45 # keep a split by 45
        grad_orientation = grad_orientation % 360  # Ensure [0, 360) range

        # Step 2: Non-maximum suppression
        # TODO: Apply directional filters to detect edge direction responses
        # Hint: directional_filter outputs 8 directional responses
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        thin_edges = grad_magnitude.clone()
        
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) | (positive_idx == neg_i)
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (~is_max) & is_oriented_i
            thin_edges[to_remove] = 0.0

        # Step 3: Thresholding
        if low_threshold is not None:
            # TODO: Create a binary mask of edges above the low threshold
            low = thin_edges > low_threshold

            if high_threshold is not None:
                # TODO: Create another mask for edges above the high threshold
                high = thin_edges > high_threshold
                #get black/gray/white only
                thin_edges = low.float() * 0.5 + high.float() * 0.5

                if hysteresis:
                    # TODO: Use a convolution to connect weak edges to strong edges
                    # Hint: Convolve with hysteresis kernel and check response > 1
                    weak = (thin_edges == 0.5).float()
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high.float() * 1 + weak_is_high * 1

            else:
                # TODO: Use only the low threshold mask as the final edge map
                thin_edges = low.float() * 1

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges








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

    # Convert image to tensor format [B, C, H, W]
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    
    canny_filter = CannyFilter()
    blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges = canny_filter.forward(
        image_tensor, low_threshold=0.05, high_threshold=0.15, hysteresis=True
    )
    
    # Convert back to numpy for display
    thin_edges_np = thin_edges.squeeze().detach().numpy()
    thin_edges_np = (thin_edges_np * 255).astype(np.uint8)
    
    cv2.imshow("Custom Canny", thin_edges_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0
if __name__ == "__main__":
    main()