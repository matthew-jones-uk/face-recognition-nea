from skimage.feature import hog
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from multiprocessing import Pool, Queue
from functools import partial
from os.path import join
from os import listdir
import logging
import skimage

import warnings
warnings.filterwarnings('ignore')

class Face():
    def __init__(self, top_left, bottom_right, probability):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.probability = probability
        self.face_image = None

    def find_face_image(self, image):
        self.face_image = image[self.top_left[0]:self.bottom_right[0],self.top_left[1]:self.bottom_right[1]]
        return self.face_image

class DetectionOptions():
    def __init__(self, factor_difference=0.5, overlap_percentage=0.1, 
                 accept_threshold=0.9, minimum_factor=1):
        self.factor_difference = factor_difference
        self.overlap_percentage = overlap_percentage
        self.accept_threshold = accept_threshold
        self.minimum_factor = minimum_factor

class HOGOptions():
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, window_size=(64, 64)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.nbins = nbins
        self.window_size = window_size

class TrainingOptions():
    def __init__(self, testing_proportion=0.15, equalise=True):
        self.testing_proportion = testing_proportion
        self.equalise = equalise

class Model():
    def __init__(self, svm_model, accuracy, hog_options):
        self.svm_model = svm_model
        self.accuracy = accuracy
        self.hog_options = hog_options

def load_image(path):
    return imread(path)

def _sliding_window(image, model, scale, hog_options=HOGOptions(), detection_options=DetectionOptions()):
    '''A sliding window worker method to find faces using image pyramid scales.
    
    Args:
        image (list): A black and white list represented image to be scanned.
        model (LinearSVM): An sklearn linear svm model with proba trained with hog data.
        hog_options (HOGOptions): Defaults to HOGOptions().
        detection_options (DetectionOptions): Defaults to DetectionOptions().
        scale (double): Scale of the image.

    Returns:
        All suspected faces as a list.
    '''
    found_faces = list()
    # Rescale the image to form an image pyramid.
    rescaled_image = rescale(image, 1/scale, mode='reflect')
    # Calculate overlap pixel size from dimension average.
    overlap_amount = int(sum(rescaled_image.shape) * 0.5 * detection_options.overlap_percentage)
    # Go through the vertical and horizontal pixels to form a sliding window.
    for row_start in range(0, rescaled_image.shape[0] - hog_options.window_size[0], overlap_amount):
        for column_start in range(0, rescaled_image.shape[1] - hog_options.window_size[1], overlap_amount):
            row_end = row_start + hog_options.window_size[0]
            column_end = column_start + hog_options.window_size[1]
            window = rescaled_image[row_start:row_end, column_start:column_end]  # Crop the desired window from the image.
            window_hog = hog(window,orientations=hog_options.orientations, pixels_per_cell=hog_options.pixels_per_cell,
                        cells_per_block=hog_options.cells_per_block)  # Calculate the Histogram of Orientate Gradients for the desired window.
            model_probability = model.predict_proba([window_hog])[:,1][0]  # Calculate the probability of the window containing a face.
            if model_probability >= detection_options.accept_threshold:
                # Scale the window coordinates to the full size image
                found_faces.append(Face((int(row_start*scale),int(column_start*scale)),(int(row_end*scale),int(column_end*scale)),model_probability))
    return found_faces

def find_all_face_boxes(image, complete_model, detection_options=DetectionOptions()):
    '''A function to create _sliding_window() processes to detect faces.

    Args:
        image (list): An list represented image to scan.
        complete_model (Model): An object containing a sklearn LinearSVM model with proba and the HOG configuration it was trained with.
        detection_options (DetectionOptions): Defaults to DetectionOptions().

    Returns:
        list: A list containing detected faces in the form of Face objects.
    '''
    model = complete_model.svm_model
    hog_options = complete_model.hog_options

    if image.shape[0] < hog_options.window_size[0] or image.shape[1] < hog_options.window_size[1]:
        print('Image is too small for set window size')
    # Calculate the maximum factor that the window can be multiplied by according to the size of the image.
    if image.shape[0] < image.shape[1]:
        maximum_factor = image.shape[0] / hog_options.window_size[0]
    else:
        maximum_factor = image.shape[1] / hog_options.window_size[1]
    image = rgb2gray(image)
    scales = list()
    scale_factor = detection_options.minimum_factor
    while scale_factor <= maximum_factor:
        scales.append(scale_factor)
        scale_factor += detection_options.factor_difference
    # No maximum processes is defined so it will default to the number of CPU cores.
    pool = Pool()
    # Creates a partial object so that the pool map can properly pass arguments to the sliding window function.
    function = partial(_sliding_window, image, model, hog_options=hog_options, detection_options=detection_options)
    # Uses the worker processes to run the sliding window function.
    possible_faces = pool.map(function, scales)
    pool.close()
    pool.join()
    # Flatten the list
    possible_faces = [face for scale_list in possible_faces for face in scale_list]
    return possible_faces

def generate_hog_data(image, hog_options=HOGOptions()):
    if image.shape != hog_options.window_size:
        print('Resizing this could potentially lead to bad data')
        image = resize(image,hog_options.window_size)
    hog_image = hog(image, orientations=hog_options.orientations, pixels_per_cell=hog_options.pixels_per_cell,
        cells_per_block=hog_options.cells_per_block)  # Calculate the Histogram of Orientate Gradients for the desired window.
    return hog_image

def generate_hog_data_from_dir(folder_path, hog_options=HOGOptions(), limit=None):
    images = listdir(path=folder_path)
    if limit:
        images = images[:limit]
    hog_data = list()
    for image_name in images:
        try:
            image = imread(join(folder_path,image_name), as_gray=True)
            hog_image = generate_hog_data(image, hog_options=hog_options)
            hog_data.append(hog_image)
        except FileNotFoundError:
            print('Failed to load '+image_name)
    return hog_data

def _calculate_equalise(x, y):
    return min(x, y)

def split_training_data(x, y, test_percentage=0.2, equalise=True):
    if equalise and len(x) != len(y):
        max_value = _calculate_equalise(len(x),len(y))
        x = x[:max_value]
        y = y[:max_value]
    x_test_size = int(len(x)*test_percentage)
    y_test_size = int(len(y)*test_percentage)
    x_test = x[:x_test_size]
    y_test = y[:y_test_size]
    x = x[x_test_size:]
    y = y[y_test_size:]
    return x, y, x_test, y_test

def premade_train(positive_train, negative_train, positive_test, negative_test):
    """Similar to the train() function, however this is for when users have
       already split their data, done any equalisation and extraced features.
    
    Args:
        positive_train (list): A list of training images that are HOG features of faces.
        negative_train (list): A list of training images that are HOG features of not faces.
        positive_test (list): A list of testing images that are HOG features of faces.
        negative_test (list): A list of testing images that are HOG features of not faces.
    
    Returns:
        LinearSVM: just the completed SVM model.
        float: the accuracy of the model.
    """

    x = list()
    y = list()
    x_test = list()
    y_test = list()
    for data in positive_train:
        x.append(data)
        y.append(1)
    for data in negative_train:
        x.append(data)
        y.append(0)
    for data in positive_test:
        x_test.append(data)
        y_test.append(1)
    for data in negative_test:
        x_test.append(data)
        y_test.append(0)
    parameter_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(LinearSVC(), parameter_grid)
    grid_search.fit(x, y)
    svm = grid_search.best_estimator_
    classifier = CalibratedClassifierCV(svm)
    classifier.fit(x, y)
    score = round(classifier.score(x_test, y_test) * 100)
    return classifier, score

def train(positive_faces_filepath, negative_faces_filepath, hog_options=HOGOptions(), training_options=TrainingOptions()):
    positive_hog = generate_hog_data_from_dir(positive_faces_filepath, hog_options=hog_options)
    negative_hog = generate_hog_data_from_dir(negative_faces_filepath, hog_options=hog_options)
    pos_train, neg_train, pos_test, neg_test = split_training_data(positive_hog, negative_hog,
        test_percentage=training_options.testing_proportion, equalise=training_options.equalise)
    svm_model, score = premade_train(pos_train, neg_train, pos_test, neg_test)
    model = Model(svm_model, round(score), hog_options)
    return model
