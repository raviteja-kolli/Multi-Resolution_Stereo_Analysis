import cv2
import numpy as np
import sys
import image_blending as ib
import region_based_analysis as rba
import feature_based_analysis as fba


def multi_resolution_analysis(image1, image2, template_size, window_size, n, search_both=False, matching_score="SAD", feature_based=False):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param n: number of levels to use in the image pyramid
    :param search_both: whether or not to search in both directions. By default it will not.
    :param matching_score: "SAD", "SSD", "NCC"
    :param feature_based: whether or not to use feature-based analysis. By default it will be region-based.
    :return: the disparity map result from the multi-resolution analysis of the 2 images
    """
    image1_pyramid = ib.gaussian_pyramid(image1, n)
    image2_pyramid = ib.gaussian_pyramid(image2, n)
    starting_disparity = None
    for k in range(n-1, -1, -1):
        left_to_right, right_to_left = get_disparity_maps(image1_pyramid[k], image2_pyramid[k], template_size,
                                                          window_size, n, search_both, matching_score, starting_disparity, feature_based)
        assert(np.size(left_to_right, 0) == np.size(right_to_left, 0))
        assert(np.size(left_to_right, 1) == np.size(right_to_left, 1))
        # Do validity check
        for i in range(0, np.size(left_to_right, 0)):
            for j in range(0, np.size(left_to_right, 1)):
                l_to_r_value = np.rint(left_to_right[i, j]).astype(np.uint8)
                r_to_l_value = np.rint(right_to_left[i, j+l_to_r_value]).astype(np.uint8) if j+l_to_r_value < np.size(left_to_right, 1) else np.size(left_to_right, 1)
                if l_to_r_value != r_to_l_value:
                    left_to_right[i, j] = 0
        # Average out the zeroes
        left_to_right = average_out_zeroes(left_to_right)
        starting_disparity = ib.expand(left_to_right)
    return left_to_right


def average_out_zeroes(disparity_map):
    """
    :param disparity_map: a disparity map of varying size
    :return: The disparity map with zeroes replaced by the average of their (5x5) neighborhood
    """
    height = np.size(disparity_map, 0)
    width = np.size(disparity_map, 1)
    for i in range(0, height):
        for j in range(0, width):
            if disparity_map[i, j] == 0:
                i_start = i - 1 if i > 0 else 0
                i_end = i + 1 if i < height else height
                j_start = j - 1 if j > 0 else 0
                j_end = j + 1 if j < width else width
                neighborhood = disparity_map[i_start:i_end, j_start:j_end]
                avg = np.sum(neighborhood)/9
                disparity_map[i, j] = avg
    return disparity_map


def get_disparity_maps(image1, image2, template_size, window_size, n,
                       search_both=False, matching_score="SAD", starting_disparity=None, feature_based=False):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param n: number of levels to use in the image pyramid
    :param search_both: whether or not to search in both directions. By default it will not.
    :param matching_score: "SAD", "SSD", "NCC"
    :param feature_based: whether or not to use feature-based analysis. By default it will be region-based.
    :param starting_disparity: disparity map to use for guiding the search
    :return: left_to_right and right_to_left disparity maps in a tuple
    """
    if matching_score == "SAD":
        if search_both:
            left_to_right = rba.region_based_analysis_sad(image1, image2, template_size, window_size,
                                                          search_direction="BOTH", starting_disparity=starting_disparity)
            right_to_left = rba.region_based_analysis_sad(image2, image1, template_size, window_size,
                                                          search_direction="BOTH", starting_disparity=starting_disparity)
            if feature_based:
                left_to_right = fba.feature_based_analysis_sad(image1, image2, (7, 7), 50,
                                                               left_to_right, search_direction="BOTH")
                right_to_left = fba.feature_based_analysis_sad(image2, image1, (7, 7), 50,
                                                               right_to_left, search_direction="BOTH")
        else:
            left_to_right = rba.region_based_analysis_sad(image1, image2, template_size, window_size,
                                                          search_direction="R", starting_disparity=starting_disparity)
            right_to_left = rba.region_based_analysis_sad(image2, image1, template_size, window_size,
                                                          search_direction="L", starting_disparity=starting_disparity)
    elif matching_score == "SSD":
        if search_both:
            left_to_right = rba.region_based_analysis_ssd(image1, image2, template_size, window_size,
                                                          search_direction="BOTH")
            right_to_left = rba.region_based_analysis_ssd(image2, image1, template_size, window_size,
                                                          search_direction="BOTH")
            if feature_based:
                left_to_right = fba.feature_based_analysis_ssd(image1, image2, (7, 7), 50,
                                                               left_to_right, search_direction="BOTH")
                right_to_left = fba.feature_based_analysis_ssd(image2, image1, (7, 7), 50,
                                                               right_to_left, search_direction="BOTH")
        else:
            left_to_right = rba.region_based_analysis_ssd(image1, image2, template_size, window_size,
                                                          search_direction="R")
            right_to_left = rba.region_based_analysis_ssd(image2, image1, template_size, window_size,
                                                          search_direction="L")
    elif matching_score == "NCC":
        if search_both:
            left_to_right = rba.region_based_analysis_ncc(image1, image2, template_size, window_size,
                                                          search_direction="BOTH")
            right_to_left = rba.region_based_analysis_ncc(image2, image1, template_size, window_size,
                                                          search_direction="BOTH")
            if feature_based:
                left_to_right = fba.feature_based_analysis_ncc(image1, image2, (7, 7), 50,
                                                               left_to_right, search_direction="BOTH")
                right_to_left = fba.feature_based_analysis_ncc(image2, image1, (7, 7), 50,
                                                               right_to_left, search_direction="BOTH")
        else:
            left_to_right = rba.region_based_analysis_ncc(image1, image2, template_size, window_size,
                                                          search_direction="R")
            right_to_left = rba.region_based_analysis_ncc(image2, image1, template_size, window_size,
                                                          search_direction="L")
    else:
        print("ERROR: invalid matching score provided")
        return None
    return left_to_right, right_to_left