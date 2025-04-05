import cv2
import numpy as np
import sys


def region_based_analysis_sad(image1, image2, template_size, window_size, search_direction="BOTH", starting_disparity=None):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param search_direction: either "R", "L", or "BOTH"
    :param starting_disparity: initial disparity map to use for guiding search. None by default
    :return: the disparity map result from the region-based analysis of the two images using the SAD matching score
    """
    # images should be of the same size
    assert np.size(image1) == np.size(image2)
    image1_greyscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_greyscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    assert np.size(image1_greyscale) == np.size(image2_greyscale)
    image_height = np.size(image1_greyscale, 0)
    image_width = np.size(image1_greyscale, 1)
    template_width = template_size[0]
    template_height = template_size[1]
    template_pixels = template_height*template_width
    disparity_map = np.zeros((image_height, image_width), dtype=np.float32)
    
    # for every pixel where a full template chunk can be taken
    for i in range(0, image_height-template_height):
        for j in range(0, image_width-template_width):
            min_sad_score = sys.maxsize
            min_sad_location = None
            template = image1_greyscale[i:i+template_height, j:j+template_width]
            # Subtract the templates mean from it
            template_mean = np.sum(template) / template_pixels
            template = template - template_mean
            left_boundary, right_boundary = calculate_search_window_region_based(j, window_size, template_width, image_width, search_direction)
            
            if starting_disparity is not None:
                init_disp_value = np.rint(starting_disparity[i, j]).astype(np.uint8)
            else:
                init_disp_value = 0
            distance = 0
            
            for k in range(left_boundary, right_boundary):
                # NEW
                if distance == j-init_disp_value and search_direction == "BOTH":
                    if k + init_disp_value * 2 < right_boundary:
                        k += init_disp_value*2
                elif distance == j-init_disp_value and search_direction == "L":
                    if k + init_disp_value < right_boundary:
                        k += init_disp_value
                elif distance == init_disp_value and search_direction == "R":
                    if k + init_disp_value < right_boundary:
                        k += init_disp_value
                image2_matrix = image2_greyscale[i:i+template_height, k:k+template_width]
                # subtract the matrix's mean from it
                image2_matrix_mean = np.sum(image2_matrix) / template_pixels
                image2_matrix = image2_matrix - image2_matrix_mean
                sad_matrix = abs(template - image2_matrix)
                total_sad = np.sum(sad_matrix)
                if total_sad < min_sad_score:
                    min_sad_score = total_sad
                    min_sad_location = distance
                distance += 1
            disparity_value = abs(window_size - min_sad_location) if min_sad_location is not None else 0
            disparity_matrix = np.zeros((template_height, template_width))
            disparity_matrix.fill(disparity_value)
            disparity_map[i:i+template_height, j:j+template_width] = disparity_matrix

    return disparity_map


def region_based_analysis_ssd(image1, image2, template_size, window_size, search_direction="BOTH", starting_disparity=None):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param search_direction: either "R", "L", or "BOTH"
    :param starting_disparity: initial disparity map to use for guiding search. None by default
    :return: the disparity map result from the region-based analysis of the two images using the SSD matching score
    """
    # images should be of the same size
    assert np.size(image1) == np.size(image2)
    image1_greyscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_greyscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    assert np.size(image1_greyscale) == np.size(image2_greyscale)
    image_height = np.size(image1_greyscale, 0)
    image_width = np.size(image1_greyscale, 1)
    template_width = template_size[0]
    template_height = template_size[1]
    template_pixels = template_height*template_width
    disparity_map = np.zeros((image_height, image_width), dtype=np.float32)
    # for every pixel where a full template chunk can be taken
    for i in range(0, image_height-template_height):
        for j in range(0, image_width-template_width):
            min_ssd_score = sys.maxsize
            min_ssd_location = None
            template = image1_greyscale[i:i+template_height, j:j+template_width]
            # subtrack the templates mean from it
            template_mean = np.sum(template) / template_pixels
            template = template - template_mean
            left_boundary, right_boundary = calculate_search_window_region_based(j, window_size, template_width, image_width, search_direction)
            if starting_disparity is not None:
                init_disp_value = np.rint(starting_disparity[i, j]).astype(np.uint8)
            else:
                init_disp_value = 0
            distance = 0
            for k in range(left_boundary, right_boundary):
                # NEW
                if distance == j-init_disp_value and search_direction == "BOTH":
                    if k + init_disp_value * 2 < right_boundary:
                        k += init_disp_value*2
                elif distance == j-init_disp_value and search_direction == "L":
                    if k + init_disp_value < right_boundary:
                        k += init_disp_value
                elif distance == init_disp_value and search_direction == "R":
                    if k + init_disp_value < right_boundary:
                        k += init_disp_value
                image2_matrix = image2_greyscale[i:i+template_height, k:k+template_width]
                # subtract the matrix's mean from it
                image2_matrix_mean = np.sum(image2_matrix) / template_pixels
                image2_matrix = image2_matrix - image2_matrix_mean
                ssd_matrix = (template-image2_matrix)**2
                total_ssd = np.sum(ssd_matrix)
                if total_ssd < min_ssd_score:
                    min_ssd_score = total_ssd
                    min_ssd_location = distance
                distance += 1
            disparity_value = abs(window_size - min_ssd_location) if min_ssd_location is not None else 0
            disparity_matrix = np.zeros((template_height, template_width))
            disparity_matrix.fill(disparity_value)
            disparity_map[i:i+template_height, j:j+template_width] = disparity_matrix

    return disparity_map


def region_based_analysis_ncc(image1, image2, template_size, window_size, search_direction="BOTH", starting_disparity=None):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param search_direction: either "R", "L", or "BOTH"
    :param starting_disparity: initial disparity map to use for guiding search. None by default
    :return: the disparity map result from the region-based analysis of the two images using the NCC matching score
    """
    # images should be of the same size
    assert np.size(image1) == np.size(image2)
    image1_greyscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_greyscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    assert np.size(image1_greyscale) == np.size(image2_greyscale)
    image_height = np.size(image1_greyscale, 0)
    image_width = np.size(image1_greyscale, 1)
    template_width = template_size[0]
    template_height = template_size[1]
    template_pixels = template_height*template_width
    disparity_map = np.zeros((image_height, image_width), dtype=np.float32)
    # for every pixel where a full template chunk can be taken
    for i in range(0, image_height-template_height):
        for j in range(0, image_width-template_width):
            max_ncc_score = -1
            max_ncc_location = None
            template = image1_greyscale[i:i+template_height, j:j+template_width]
            # subtrack the templates mean from it
            template_mean = np.sum(template) / template_pixels
            template = template - template_mean
            left_boundary, right_boundary = calculate_search_window_region_based(j, window_size, template_width, image_width, search_direction)
            if starting_disparity is not None:
                init_disp_value = np.rint(starting_disparity[i, j]).astype(np.uint8)
            else:
                init_disp_value = 0
            distance = 0
            for k in range(left_boundary, right_boundary):
                # NEW
                if distance == j-init_disp_value and search_direction == "BOTH":
                    if k + init_disp_value * 2 < right_boundary:
                        k += init_disp_value*2
                elif distance == j-init_disp_value and search_direction == "L":
                    if k + init_disp_value < right_boundary:
                        k += init_disp_value
                elif distance == init_disp_value and search_direction == "R":
                    if k + init_disp_value < right_boundary:
                        k += init_disp_value
                image2_matrix = image2_greyscale[i:i+template_height, k:k+template_width]
                # subtract the matrix's mean from it
                image2_matrix_mean = np.sum(image2_matrix) / template_pixels
                image2_matrix = image2_matrix - image2_matrix_mean
                denominator = (np.sum(template**2)*np.sum(image2_matrix**2))**0.5
                total_ncc = np.sum(template*image2_matrix)/denominator
                assert(-1 <= total_ncc <= 1)
                if total_ncc > max_ncc_score:
                    max_ncc_score = total_ncc
                    max_ncc_location = distance
                distance += 1
            disparity_value = abs(window_size - max_ncc_location) if max_ncc_location is not None else 0
            disparity_matrix = np.zeros((template_height, template_width))
            disparity_matrix.fill(disparity_value)
            disparity_map[i:i+template_height, j:j+template_width] = disparity_matrix

    return disparity_map


def calculate_search_window_region_based(j, window_size, template_width, image_width, search_direction):
    """
    :param j: column of the current pixel
    :param window_size: length of search window
    :param template_width: width of the template
    :param image_width: width of the image
    :param search_direction: R, L or BOTH
    :return: left_boundary and right_boundary of the window to search in a tuple
    """
    if search_direction == "R":
        left_boundary = j
        right_boundary = j + window_size if j + window_size + template_width < image_width else image_width - template_width
    elif search_direction == "L":
        left_boundary = j - window_size if j >= window_size else 0
        right_boundary = j
    elif search_direction == "BOTH":
        left_boundary = j - window_size if j >= window_size else 0
        right_boundary = j + window_size if j + window_size + template_width < image_width else image_width - template_width
    else:
        print("ERROR: invalid search direction specified")
        left_boundary = None
        right_boundary = None
    return left_boundary, right_boundary


def normalize_disparity_values(disparity_map):
    """
    :param disparity_map: a disparity map of varying size
    :return: disparity map with values normalized into the 0-255 range where closer
    things are lighter and farther things are darker
    """
    largest = disparity_map.max()
    increment = 255 / largest
    disparity_map = (255 - disparity_map) * increment
    disparity_map = np.rint(disparity_map)
    disparity_map = disparity_map.astype(np.uint8)
    disparity_map = cv2.bitwise_not(disparity_map)
    return disparity_map

