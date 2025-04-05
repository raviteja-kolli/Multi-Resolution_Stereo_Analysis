import cv2
import numpy as np
import sys


def feature_based_analysis_descriptor_value(image1, image2, template_size, window_size, starting_disparity_map, search_direction="BOTH"):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param search_direction: either "R", "L", or "BOTH"
    :return: the disparity map result from the feature-based analysis of the two images using the descriptor value as matching score
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
    new_disparity_map = starting_disparity_map
    corner_response1, corner_coordinates1 = get_harris_corners(image1_greyscale)
    corner_response2, corner_coordinates2 = get_harris_corners(image2_greyscale)
    
    # Find the region in image2 with the most similar corner response
    for coordinate in corner_coordinates1.keys():
        min_response_difference = sys.maxsize
        min_response_difference_location = None
        i = coordinate[0]
        j = coordinate[1]
        i_start = i - template_height//2 if i - template_height//2 > 0 else 0
        i_end = i + template_height//2 if i + template_height//2 < image_height else image_height
        j_start = j - template_width//2 if j - template_width//2 > 0 else 0
        j_end = j + template_width//2 if j + template_width//2 < image_width else image_width
        template = corner_response1[i_start:i_end, j_start:j_end]
        left_boundary, right_boundary = calculate_search_window_feature_based(j, window_size, template_width, image_width, search_direction)
        distance = 0
        for k in range(left_boundary, right_boundary):
            k_start = k - template_width // 2 if k - template_width // 2 > 0 else 0
            k_end = k + template_width // 2 if k + template_width // 2 < image_width else image_width
            width_dif = np.size(template, 1) - (k_end-k_start)
            k_end += width_dif
            image2_response_matrix = corner_response2[i_start:i_end, k_start:k_end]
            response_matrix = abs(template - image2_response_matrix)
            total_response_difference = np.sum(response_matrix)
            if total_response_difference < min_response_difference:
                min_response_difference = total_response_difference
                min_response_difference_location = distance
            distance += 1
            disparity_value = abs(window_size - min_response_difference_location) if min_response_difference_location is not None else 0
            disparity_matrix = np.zeros((i_end-i_start, j_end-j_start))
            disparity_matrix.fill(disparity_value)
            new_disparity_map[i_start:i_end, j_start:j_end] = disparity_matrix
    return new_disparity_map


def feature_based_analysis_sad(image1, image2, template_size, window_size, starting_disparity_map, search_direction="BOTH"):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param search_direction: either "R", "L", or "BOTH"
    :return: the disparity map result from the feature-based analysis of the two images using the SAD matching score
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
    new_disparity_map = starting_disparity_map
    corner_response1, corner_coordinates1 = get_harris_corners(image1_greyscale)
    for coordinate in corner_coordinates1.keys():
        min_sad_score = sys.maxsize
        min_sad_location = None
        i = coordinate[0]
        j = coordinate[1]
        i_start = i - template_height//2 if i - template_height//2 > 0 else 0
        i_end = i + template_height//2 if i + template_height//2 < image_height else image_height
        j_start = j - template_width//2 if j - template_width//2 > 0 else 0
        j_end = j + template_width//2 if j + template_width//2 < image_width else image_width
        template = image1_greyscale[i_start:i_end, j_start:j_end]
        # Subtract the templates mean from it
        template_mean = np.sum(template) / template_pixels
        template = template - template_mean
        left_boundary, right_boundary = calculate_search_window_feature_based(j, window_size, template_width, image_width, search_direction)
        distance = 0
        for k in range(left_boundary, right_boundary):
            k_start = k - template_width // 2 if k - template_width // 2 > 0 else 0
            k_end = k + template_width // 2 if k + template_width // 2 < image_width else image_width
            width_dif = np.size(template, 1) - (k_end-k_start)
            k_end += width_dif
            image2_matrix = image2_greyscale[i_start:i_end, k_start:k_end]
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
            disparity_matrix = np.zeros((i_end-i_start, j_end-j_start))
            disparity_matrix.fill(disparity_value)
            new_disparity_map[i_start:i_end, j_start:j_end] = disparity_matrix
    return new_disparity_map


def feature_based_analysis_ssd(image1, image2, template_size, window_size, starting_disparity_map, search_direction="BOTH"):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param search_direction: either "R", "L", or "BOTH"
    :return: the disparity map result from the feature-based analysis of the two images using the ssd matching score
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
    new_disparity_map = starting_disparity_map
    corner_response1, corner_coordinates1 = get_harris_corners(image1_greyscale)
    for coordinate in corner_coordinates1.keys():
        min_ssd_score = sys.maxsize
        min_ssd_location = None
        i = coordinate[0]
        j = coordinate[1]
        i_start = i - template_height//2 if i - template_height//2 > 0 else 0
        i_end = i + template_height//2 if i + template_height//2 < image_height else image_height
        j_start = j - template_width//2 if j - template_width//2 > 0 else 0
        j_end = j + template_width//2 if j + template_width//2 < image_width else image_width
        template = image1_greyscale[i_start:i_end, j_start:j_end]
        # Subtract the templates mean from it
        template_mean = np.sum(template) / template_pixels
        template = template - template_mean
        left_boundary, right_boundary = calculate_search_window_feature_based(j, window_size, template_width, image_width, search_direction)
        distance = 0
        for k in range(left_boundary, right_boundary):
            k_start = k - template_width // 2 if k - template_width // 2 > 0 else 0
            k_end = k + template_width // 2 if k + template_width // 2 < image_width else image_width
            width_dif = np.size(template, 1) - (k_end-k_start)
            k_end += width_dif
            image2_matrix = image2_greyscale[i_start:i_end, k_start:k_end]
            # subtract the matrix's mean from it
            image2_matrix_mean = np.sum(image2_matrix) / template_pixels
            image2_matrix = image2_matrix - image2_matrix_mean
            ssd_matrix = (template - image2_matrix) ** 2
            total_ssd = np.sum(ssd_matrix)
            if total_ssd < min_ssd_score:
                min_ssd_score = total_ssd
                min_ssd_location = distance
            distance += 1
            disparity_value = abs(window_size - min_ssd_location) if min_ssd_location is not None else 0
            disparity_matrix = np.zeros((i_end-i_start, j_end-j_start))
            disparity_matrix.fill(disparity_value)
            new_disparity_map[i_start:i_end, j_start:j_end] = disparity_matrix
    return new_disparity_map


def feature_based_analysis_ncc(image1, image2, template_size, window_size, starting_disparity_map, search_direction="BOTH"):
    """
    :param image1: an image of varying size (numpy ndarray)
    :param image2: an image of varying size (numpy ndarray) that is a view of the same scene
    with the camera further to the right
    :param template_size: tuple of the form (width, height) representing the dimensions of the matching window
    :param window_size: number of pixels to the right and left within which to search for a template match
    :param search_direction: either "R", "L", or "BOTH"
    :return: the disparity map result from the feature-based analysis of the two images using the ncc matching score
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
    new_disparity_map = starting_disparity_map
    corner_response1, corner_coordinates1 = get_harris_corners(image1_greyscale)
    for coordinate in corner_coordinates1.keys():
        max_ncc_score = -1
        max_ncc_location = None
        i = coordinate[0]
        j = coordinate[1]
        i_start = i - template_height//2 if i - template_height//2 > 0 else 0
        i_end = i + template_height//2 if i + template_height//2 < image_height else image_height
        j_start = j - template_width//2 if j - template_width//2 > 0 else 0
        j_end = j + template_width//2 if j + template_width//2 < image_width else image_width
        template = image1_greyscale[i_start:i_end, j_start:j_end]
        # Subtract the templates mean from it
        template_mean = np.sum(template) / template_pixels
        template = template - template_mean
        left_boundary, right_boundary = calculate_search_window_feature_based(j, window_size, template_width, image_width, search_direction)
        distance = 0
        for k in range(left_boundary, right_boundary):
            k_start = k - template_width//2 if k - template_width//2 > 0 else 0
            k_end = k + template_width//2 if k + template_width//2 < image_width else image_width
            width_dif = np.size(template, 1) - (k_end-k_start)
            k_end += width_dif
            image2_matrix = image2_greyscale[i_start:i_end, k_start:k_end]
            # subtract the matrix's mean from it
            image2_matrix_mean = np.sum(image2_matrix) / template_pixels
            image2_matrix = image2_matrix - image2_matrix_mean
            denominator = (np.sum(template ** 2) * np.sum(image2_matrix ** 2)) ** 0.5
            total_ncc = np.sum(template * image2_matrix) / denominator
            assert (-1 <= total_ncc <= 1)
            if total_ncc > max_ncc_score:
                max_ncc_score = total_ncc
                max_ncc_location = distance
            distance += 1
            disparity_value = abs(window_size - max_ncc_location) if max_ncc_location is not None else 0
            disparity_matrix = np.zeros((i_end-i_start, j_end-j_start))
            disparity_matrix.fill(disparity_value)
            new_disparity_map[i_start:i_end, j_start:j_end] = disparity_matrix
    return new_disparity_map


def calculate_search_window_feature_based(j, window_size, template_width, image_width, search_direction):
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
        right_boundary = j + window_size if j + window_size + template_width//2 < image_width else image_width - template_width//2
    elif search_direction == "L":
        left_boundary = j - window_size if j >= window_size else 0
        right_boundary = j
    elif search_direction == "BOTH":
        left_boundary = j - window_size if j >= window_size else 0
        right_boundary = j + window_size if j + window_size + template_width//2 < image_width else image_width - template_width//2
    else:
        print("ERROR: invalid search direction specified")
        left_boundary = None
        right_boundary = None
    return left_boundary, right_boundary


def get_harris_corners(image_greyscale):
    """
    :param image_greyscale: a greyscale image of varying size
    :return: a dictionary where keys are tuples of the form (i, j) and values are the corner
    response measure from the harris detector at those (i, j) pixel coordinates
    """
    corner_response = cv2.cornerHarris(image_greyscale, 2, 3, 0.04)
    corner_response = cv2.dilate(corner_response, None)
    image_height = np.size(image_greyscale, 0)
    image_width = np.size(image_greyscale, 1)
    assert (np.size(corner_response, 0) == image_height)
    assert (np.size(corner_response, 1) == image_width)
    corner_coordinates = {}
    for i in range(0, image_height):
        for j in range(0, image_width):
            if corner_response[i, j] > 0.01 * corner_response.max():
                corner_coordinates.update({(i, j) : corner_response[i, j]})
    return corner_response, corner_coordinates

