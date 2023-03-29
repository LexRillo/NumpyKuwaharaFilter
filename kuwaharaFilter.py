import numpy as np
from scipy.signal import convolve2d
from skimage import io
from skimage.color import rgba2rgb, rgb2hsv, hsv2rgb
from skimage.filters import gaussian
from skimage.util import img_as_ubyte
from pathlib import Path
import matplotlib.pyplot as plt

MIN_KERNEL_SIZE = 5
VERBOSITY = True

## Original Kuwahara filter - The idea was to make a denoising algorithm for medical imaging but it has become stylistic one where images tend to look like they were painted
## https://en.wikipedia.org/wiki/Kuwahara_filter
## Based on the Matlab original implementation of Luca Balbi 2007
## https://uk.mathworks.com/matlabcentral/fileexchange/15027-faster-kuwahara-filter
def original_Kuwahara_filter_grayscale(original_image, window_size : int):
    if VERBOSITY:
        print("Processing Original Kuwahara for grayscale image with a window size of", window_size)
    if original_image.ndim > 2:
        raise Exception ("Function requires grayscale image")
       
    kernels = create_overlapping_box_kernels(window_size)
    
    img_squared = original_image**2
    avgs = np.zeros([original_image.shape[0],original_image.shape[1], 4])
    stddevs = np.zeros([original_image.shape[0],original_image.shape[1], 4])
    
    for k in range(4):
        avgs[:, :, k] = convolve2d(original_image, kernels[k],mode='same')
        stddevs[:, :, k] = convolve2d(img_squared, kernels[k],mode='same')
        stddevs[:, :, k] = stddevs[:, :,k]-avgs[:, :, k]**2 
        
    indices = np.argmin(stddevs,2)
    
    filtered = np.zeros(original_image.shape)
    for row in range(original_image.shape[0]):
        for col in range(original_image.shape[1]):
            filtered[row,col] = avgs[row,col, indices[row,col]]
    
    return filtered
    
    
## Kuwahara filter extended to colour images naively. 
## Unfortunately due to the segmented statistical nature of filter it cannot be used as is in RGB since it causes artifacts and blurrier edges
def naive_original_Kuwahara_filter_colour(original_image, window_size : int):
    if VERBOSITY:
        print("Processing Naive Original Kuwahara for colour image with a window size of", window_size)
    if original_image.ndim ==2:
        raise Exception ("Function requires colour image")
        
    kernels = create_overlapping_box_kernels(window_size)    
    
    img_copy = original_image.copy()
    
    n_channels = original_image.shape[2]
    for channel in range(n_channels):
        image = img_copy[:, :, channel]
        img_squared = image**2
        
        avgs = np.zeros([image.shape[0],image.shape[1], 4])
        stddevs = np.zeros([image.shape[0],image.shape[1], 4])
        
        for k in range(4):
            avgs[:, :, k] = convolve2d(image, kernels[k],mode='same')
            stddevs[:, :, k] = convolve2d(img_squared, kernels[k],mode='same')
            stddevs[:, :, k] = stddevs[:, :, k]-avgs[:, :, k]**2 
            
        indices = np.argmin(stddevs,2)
        
        filtered = np.zeros(image.shape)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                filtered[row,col] = avgs[row,col, indices[row,col]]
        img_copy[:, :, channel] = filtered

    return img_copy


## Kuwahara Filter extended for colour images using HSV
def original_Kuwahara_filter_colour(original_image, window_size : int):
    if VERBOSITY:
        print("Processing Original Kuwahara for colour image with a window size of", window_size)
    if original_image.ndim ==2:
        raise Exception ("Function requires colour image")
    
    kernels = create_overlapping_box_kernels(window_size)    
    
    if original_image.shape[2] == 4:
        img_copy = rgb2hsv(rgba2rgb(original_image))
    else:
        img_copy = rgb2hsv(original_image)
    
    hue_component = img_copy[:, :, 0]
    saturation_component = img_copy[:, :, 1]
    brightness_component = img_copy[:, :, 2]
    img_squared = brightness_component**2
    
    hue_avg = np.zeros([img_copy.shape[0],img_copy.shape[1], 4])
    saturation_avg = np.zeros([img_copy.shape[0],img_copy.shape[1], 4])
    brightness_avgs = np.zeros([img_copy.shape[0],img_copy.shape[1], 4])
    stddevs = np.zeros([img_copy.shape[0],img_copy.shape[1], 4])
    
    for k in range(4):
        hue_avg[:, :, k] = convolve2d(hue_component, kernels[k],mode='same')
        saturation_avg[:, :, k] = convolve2d(saturation_component, kernels[k],mode='same')
        brightness_avgs[:, :, k] = convolve2d(brightness_component, kernels[k],mode='same')
        stddevs[:, :, k] = convolve2d(img_squared, kernels[k],mode='same')
        stddevs[:, :, k] = stddevs[:, :, k]-brightness_avgs[:, :, k]**2 
        
    indices = np.argmin(stddevs,2)
    
    filtered = np.zeros(img_copy.shape)
    for row in range(img_copy.shape[0]):
        for col in range(img_copy.shape[1]):
            filtered[row,col,0] = hue_avg[row,col, indices[row,col]]
            filtered[row,col,1] = saturation_avg[row,col, indices[row,col]]
            filtered[row,col,2] = brightness_avgs[row,col, indices[row,col]]
    
    
    img_copy = hsv2rgb(filtered)

    return img_copy

    
# TODO: Not fully working yet
## From the original paper "Artistic Edge and Corner Enhancing Smoothing" by Papari et al.
## https://www.cs.rug.nl/~imaging/artisticsmoothing/TIP_artistic.pdf
def generalized_Kuwahara_filter(original_image, window_size : int, sharpness : float = 0.5):
    ## Changes to the original Kuwahara implementation
    # change square kernel into circle kernel
    # change the four sectors into 8 sectors
    # change weighting from standard deviation
    # use the standard deviation to calculate the colour
      
    if VERBOSITY:
        print("Processing Generalized Kuwahara for colour image with a window size of", window_size)
    if original_image.ndim ==2:
        raise Exception ("Function requires colour image")
        
    kernels = create_sector_kernels(window_size)

    img_copy = original_image.astype(np.float64)
    
    img_squared = img_copy**2
    
    red_component = original_image[:,:,0]
    green_component = original_image[:,:,1]
    blue_component = original_image[:,:,2]
    
    red_component_squared = img_squared[:,:,0]
    green_component_squared = img_squared[:,:,1]
    blue_component_squared = img_squared[:,:,2]
    
    red_average = np.zeros([img_copy.shape[0],img_copy.shape[1], 8], dtype=float)
    green_average = np.zeros([img_copy.shape[0],img_copy.shape[1], 8], dtype=float)
    blue_average = np.zeros([img_copy.shape[0],img_copy.shape[1], 8], dtype=float)
    stddev_weights = np.zeros([img_copy.shape[0],img_copy.shape[1], 8], dtype=float)
    
    for k in range(8):
        red_average[:, :, k] = convolve2d(red_component, kernels[k],mode='same')
        green_average[:, :, k] = convolve2d(green_component, kernels[k],mode='same')
        blue_average[:, :, k] = convolve2d(blue_component, kernels[k],mode='same')
        red_squared_conv = convolve2d(red_component_squared, kernels[k],mode='same')
        green_squared_conv = convolve2d(green_component_squared, kernels[k],mode='same')
        blue_squared_conv = convolve2d(blue_component_squared, kernels[k],mode='same')
        red_variance = np.abs(red_squared_conv - red_average[:, :, k]**2)
        green_variance = np.abs(green_squared_conv - green_average[:, :, k]**2)
        blue_variance = np.abs(blue_squared_conv - blue_average[:, :, k]**2)
        # stddev_weights[:, :, k] = 1/(1 + np.sqrt(np.abs(stddev_weights[:, :, k]-brightness_avgs[:, :, k]**2)))
        stddev_weights[:, :, k] = (1 + red_variance + green_variance + blue_variance)**(-sharpness*0.5)
    
    filtered = np.zeros(img_copy.shape)
    filtered[:,:,0] = np.sum(red_average *stddev_weights, axis = 2)/np.sum(stddev_weights, axis = 2)
    filtered[:,:,1] = np.sum(green_average *stddev_weights, axis = 2)/np.sum(stddev_weights, axis = 2)
    filtered[:,:,2] = np.sum(blue_average *stddev_weights, axis = 2)/np.sum(stddev_weights, axis = 2)
    
    img_copy = img_as_ubyte(filtered/255)
    
    return img_copy


# TODO: implement
## From the original paper "Image and Video Abstraction by Anisotropic Kuwahara Filtering"
## http://www.umsl.edu/~kangh/Papers/kang_cgf09.pdf
def anisotropic_Kuwahara_filter():
    pass
    

# TODO: implement
## From the original paper "Anisotropic Kuwahara Filtering with Polynomial Weighting Functions"
## http://www.umsl.edu/~kangh/Papers/kang-tpcg2010.pdf
def polynomial_anisotropic_Kuwahara_filter():
    pass


# TODO: implement
## From the Matlab implementation https://uk.mathworks.com/matlabcentral/fileexchange/58260-super-fast-kuwahara-image-filter-for-n-dimensional-real-or-complex-data
def super_fast_Kuwahara_filter():
    pass

### Kernels
# Original Kawahara kernels
def create_overlapping_box_kernels(window_size : int):
    if window_size < MIN_KERNEL_SIZE or window_size%2 == 0:
        raise Exception ("Invalid argument window size")

    subkernel_shape_size = int((window_size+1)/2)
    subkernel = np.ones((subkernel_shape_size, subkernel_shape_size))/(subkernel_shape_size**2)
    assert(np.isclose(np.sum(subkernel), 1, rtol=1e-05, atol=1e-08, equal_nan=False))
    
    padding = window_size - subkernel_shape_size
    kernel_UL = np.pad(subkernel, ((0, padding),(0, padding)), 'constant')
    kernel_UR = np.pad(subkernel, ((0, padding),(padding, 0)), 'constant')
    kernel_LR = np.pad(subkernel, ((padding, 0),(padding, 0)), 'constant')
    kernel_LL = np.pad(subkernel, ((padding, 0),(0, padding)), 'constant')
    
    kernels = np.array([kernel_UL, kernel_UR, kernel_LR, kernel_LL])
    return kernels

# Gaussian kernel
def gaussian_kernel(window_size : int, sigma = 2):
    if window_size < MIN_KERNEL_SIZE or window_size%2 == 0:
        raise Exception ("Invalid argument window size")
    k = np.zeros((window_size, window_size))
    
    kernel_center = int((window_size)/2)
    k[kernel_center, kernel_center] = 1
    return gaussian(k, sigma=sigma)

# Generalized Kawahara kernels
def create_sector_kernels(window_size : int, sections : int = 8):
    if window_size < MIN_KERNEL_SIZE or window_size%2 == 0:
        raise Exception ("Invalid argument window size")
    
    # for k in range(8):
        # print(((2*k)+1)*np.pi/8 , ((2*k)+2)*np.pi/8)
    # r = int(np.floor((window_size)))
    # kernel = np.fromfunction(lambda x, y: np.arctan2(x-r, y-r) +np.pi, (2*r+1, 2*r+1))
    # io.imshow(kernel)
    # io.show()
    kernels = np.zeros([window_size,window_size, sections])
    
    original_gaussian_kernel = gaussian_kernel(window_size)
    
    r = int((window_size)/2)
    for k in range(8):
        section_kernel = np.fromfunction(lambda x, y: (is_point_in_circle(x, y, r))*(np.arctan2(x-r, y-r) + np.pi >= (((2*k))*np.pi/8)) * (np.arctan2(x-r, y-r)+np.pi <= (((2*k)+2)*np.pi/8)), (2*r+1, 2*r+1), dtype=int).astype(np.uint8)
        section_kernel = section_kernel*original_gaussian_kernel
        section_kernel = section_kernel/np.sum(section_kernel)
        assert(np.isclose(np.sum(section_kernel), 1, rtol=1e-05, atol=1e-08, equal_nan=False))
        kernels[:,:,k] = section_kernel
    
    return kernels
    

### Auxiliary functions        
def is_point_in_circle(x, y, r):
    return (x-r)**2 + (y-r)**2 <= r**2
    
    
if __name__ == '__main__':
    # Get path
    image_path = Path(".\\BusyStreet.png")

    # Read image
    img = io.imread(image_path)

    # Apply Original Kuwahara filter
    og_kawa_gray = original_Kuwahara_filter_grayscale(img[:, :, 0], 9)
    og_kawa2 = naive_original_Kuwahara_filter_colour(img, 9)
    og_kawa3 = original_Kuwahara_filter_colour(img, 9)

    # Apply Generalized Kuwahara filter
    g_kawa = generalized_Kuwahara_filter(img, 9)

    # Apply Anisotropic Kuwahara filter

    # Show images
    fig, axes = plt.subplots(3, 2, figsize=(7, 6))
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title("Image (original)")

    ax[1].imshow(og_kawa_gray, cmap=plt.cm.gray)
    ax[1].set_title("Original Kawahara gray")

    ax[2].imshow(og_kawa2, cmap=plt.cm.gray)
    ax[2].set_title("Naive Kawahara colour")

    ax[3].imshow(og_kawa3, cmap=plt.cm.gray)
    ax[3].set_title("Kawahara colour")
    
    ax[4].imshow(g_kawa, cmap=plt.cm.gray)
    ax[4].set_title("Generalized Kawahara colour")
    
    
    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()
    plt.show()