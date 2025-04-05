import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def convolve(I, H):
    """
    I is an image of varying size
    H is a kernel of varying size
    both should be numpy arrays
    returns:
    A new image that is the result of the convolution
    """
    # Stride length is assumed to be 1
    # axis zero is number of rows
    # axis one is number of columns
    # filter height
    filter_height = np.size(H, 0)
    # filter width
    filter_width = np.size(H, 1)
    # image depth
    image_channels = np.size(I, 2)
    # image height
    image_height = np.size(I, 0)
    # image width
    image_width = np.size(I, 1)
    padding_width = int(math.floor(filter_width-1)/2)
    padding_height = int(math.floor(filter_height-1)/2)
    padded_image = apply_padding(I, padding_height, padding_width)
    image_height_with_padding = np.size(padded_image, 0)
    image_width_with_padding = np.size(padded_image, 1)
    output_width = image_width_with_padding - filter_width + 1
    output_height = image_height_with_padding - filter_height + 1
    # The padding should cause the output image to have the same dimensions as the input image
    assert(output_width == image_width)
    assert(output_height == image_height)
    result_list = []
    for i in range(0, output_height):
        new_row_list = []
        for j in range(0, output_width):
            new_channel_list = []
            for k in range(0, image_channels):
                image_slice = padded_image[i:i+filter_height, j:j+filter_width, k]
                dot_product = np.sum(H*image_slice)
                new_channel_list.append(dot_product)
            new_row_list.append(new_channel_list)
        result_list.append(new_row_list)
    result = np.array(result_list)
    return result


def reduce(I, blur_image=True):
    """
    I is an image of varying size
    Does gaussian blurring then samples every other pixel
    returns:
    a copy of the image down sampled to be half the height and half the width
    """
    # Using the 2D kernel since it runs faster
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]],
                               dtype=np.float32)
    gaussian_kernel = (1 / 16) * gaussian_kernel
    blurred_image = I
    if blur_image:
        convolve(I, gaussian_kernel)
    # image height
    image_height = np.size(blurred_image, 0)
    # image width
    image_width = np.size(blurred_image, 1)
    result_list = []
    for i in range(0, image_height, 2):
        new_row_list = []
        for j in range(0, image_width, 2):
            new_row_list.append(blurred_image[i, j])
        result_list.append(new_row_list)
    result = np.array(result_list)
    return result


def expand(I):
    """
    I is an image of varying size
    returns:
    a copy of the image expanded to be twice the size
    """
    result = np.copy(I)
    new_width = np.size(result, 1)*2
    new_height = np.size(result, 0)*2
    return cv2.resize(result, dsize=(new_width, new_height))


def gaussian_pyramid(I, n, blur_image=True):
    """
    Creates a Gaussian pyramid of the image with n levels.
    I is an image of varying size
    n is the number of levels in the pyramid
    return:
    a list of images in the gaussian pyramid from largest to smallest.
    each image is a numpy ndarray.
    """
    g_pyramid = []
    cur_level = np.copy(I)
    g_pyramid.append(cur_level)
    for i in range(0, n-1):
        cur_level = reduce(cur_level, blur_image=blur_image)
        g_pyramid.append(cur_level)
    return g_pyramid


def laplacian_pyramid(I, n):
    """
    Creates a Laplacian pyramid for the image by taking the difference of Gaussians.
    I is an image of varying size
    n is the number of levels in the pyramid
    returns:
    a list of images in the laplacian pyramid from largest to smallest.
    each image is a numpy ndarray.
    """
    # first create the gaussian pyramid
    g_pyramid = gaussian_pyramid(I, n)
    l_pyramid = [None]*n
    # the smallest levels are the same in each pyramid
    l_pyramid[n-1] = g_pyramid[n-1]
    for i in range(0, n-1):
        expanded_image = expand(g_pyramid[i+1])
        desired_dimensions = np.shape(g_pyramid[i])
        # in case the dimensions are off by 1 from rounding
        expanded_image = match_dimensions(desired_dimensions, expanded_image)
        l_pyramid[i] = g_pyramid[i] - expanded_image
    return l_pyramid


def reconstruct(LI):
    """
    LI is a Laplacian pyramid (a list of numpy ndarray)
    returns:
    The reconstructed image formed by collapsing the given Laplacian pyramid
    """
    # loop from the smallest level of the pyramid to the largest
    n = len(LI)
    reconstructed = LI[n-1]
    for i in range(n-2, -1, -1):
        desired_dimensions = np.shape(LI[i])
        expanded_image = expand(reconstructed)
        # in case the dimensions are off by 1 from rounding
        expanded_image = match_dimensions(desired_dimensions, expanded_image)
        reconstructed = LI[i] + expanded_image
    reconstructed = np.rint(reconstructed)
    return reconstructed


def create_blended_pyramid(IA, IB, bitmask, n):
    """
    This function blends two images and returns the resulting image.
    IA is an image of varying size
    IB is an image of varying size
    n is the number of levels in the pyramids used in blending
    returns:
    the blended image that results from splining the images together
    """
    # Everything should have the same dimensions
    assert(np.size(IA) == np.size(IB) == np.size(bitmask))
    # Laplacian pyramid of image 1
    LA = laplacian_pyramid(IA, n)
    # Laplacian pyramid of image 2
    LB = laplacian_pyramid(IB, n)
    # Gaussian pyramid of the bitmask
    GS = gaussian_pyramid(bitmask, n)
    Lout = [None]*n
    for i in range(0, n):
        # For visual debugging
        # left_half = GS[i] * LA[i]
        # right_half = (1.0-GS[i])*LB[i]
        # cv2.imshow("bitmask", np.rint(255*GS[i]).astype(np.uint8))
        # cv2.imshow("left", left_half.astype(np.uint8))
        # cv2.imshow("right", right_half.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        Lout[i] = GS[i]*LA[i] + (1.0-GS[i])*LB[i]
    return Lout


def blend_images(IA, IB, bitmask, n):
    """
    This function blends 2 images using Laplacian pyramid blending
    with n levels
    IA is an image of varying size
    IB is an image of varying size
    bitmask is the bitmask to use in blending
    n is the number of pyramid levels to use
    returns:
    The blended image result
    """
    blended_pyramid = create_blended_pyramid(IA, IB, bitmask, n)
    blended_image = reconstruct(blended_pyramid)
    return blended_image.astype(np.uint8)


def create_bitmask(height, width, depth, anchor_point, blend_region_height, blend_region_width):
    """
    height is the height that the bitmask should be
    width is the width that the bitmask should be
    depth is the number of channels in the bitmask (3 for color images, 1 for greyscale)
    anchor_point is the pixel location of the top left hand corner of the region to be blended
    blend_region_height is the height of the region with 1's
    blend_region_width is the width of the region with 1's
    returns:
    a bitmask of the specific dimensions with 1's filling in the blend region
    """
    ones_region = np.ones((blend_region_height, blend_region_width, depth), dtype=np.float32)
    bitmask = np.zeros((height, width, depth), dtype=np.float32)
    bitmask[anchor_point[0]:anchor_point[0]+blend_region_height, anchor_point[1]:anchor_point[1]+blend_region_width] = ones_region
    return bitmask


def match_dimensions(desired_dimensions, I):
    """
    desired_dimensions is the dimensions that the image I should be
    I is an image of varying size
    returns:
    the resized image I to match the desired dimensions if it didn't already match them
    """
    result = I
    if desired_dimensions != np.shape(I):
        result = cv2.resize(result, dsize=(desired_dimensions[1], desired_dimensions[0]))
    return result


def apply_padding(I, padding_height, padding_width, apply_left=True, apply_right=True, apply_top=True, apply_bottom=True):
    """
    Helper function that applies zero-padding
    I is an image of varying size
    padding_height is the thickness of the padding to be adding on top and bottom
    padding_width is the thickness of the padding to be added on left and right
    returns:
    A new image (numpy ndarray) with the added zero-padding
    """
    # image depth
    image_channels = np.size(I, 2)
    # image height
    image_height = np.size(I, 0)
    # image width
    image_width = np.size(I, 1)
    zero_row = np.array([[[0]*image_channels]*image_width], dtype=np.float32)
    zero_column = np.array([[[0]*image_channels]]*(image_height+padding_height*2), dtype=np.float32)
    result = np.copy(I)
    for i in range(0, padding_height):
        if apply_top:
            result = np.concatenate((zero_row, result), axis=0)
        if apply_bottom:
            result = np.concatenate((result, zero_row), axis=0)
    for j in range(0, padding_width):
        if apply_left:
            result = np.concatenate((zero_column, result), axis=1)
        if apply_right:
            result = np.concatenate((result, zero_column), axis=1)
    return result
