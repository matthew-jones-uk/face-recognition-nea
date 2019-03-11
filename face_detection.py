from multiprocessing import Pool, Queue
from functools import partial
from os.path import join
from os import listdir
import logging
import warnings
#from skimage.feature import hog
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from hog import HOGOptions
from hog import hog

# Disable unnecessary sklearn warning spam.
warnings.filterwarnings('ignore')

class Face():
    '''Face object to store information on a detected face.
    Args:
        top_left (tuple(integer, integer)): y and x index values of the top left of the face.
        bottom_right (tuple(integer, integer)): y and x valiues of the bottom left of the face.
        probability (float): Probability that the detected face is a face.
        face_image (numpy.array, optional): Defaults to None. Numpy array of cropped face image.
    '''
    def __init__(self, top_left, bottom_right, probability, face_image=None):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.probability = probability
        self.face_image = face_image

    def find_face_image(self, image):
        '''Find the face and crop given the full image and store the image inside object. Not done
           by default to preserve memory usage.
        Args:
            image (numpy.array): Full image containing face in numpy array form.
        Returns:
            face_image (numpy.array): Cropped image of face in numpy array form.
        '''
        self.face_image = image[self.top_left[0]:self.bottom_right[0],self.top_left[1]:self.bottom_right[1]]
        return self.face_image

class DetectionOptions():
    '''Detection options object to store detection configuration that's not for the HOG.
    Args:
        factor_difference (float): Defaults to 0.5. The factor to increase by for image pyramid.
        overlap_percentage (float): Defaults to 0.1. The percentage of the sliding window to
                                    move by each time.
        accept_threshold (float): Defaults to 0.7. The confidence threshold to accept an image as
                                  a face.
        minimum_factor (float): Defaults to 1.0. The default factor of a window to start the image
                                pyramid at
    '''
    def __init__(self, factor_difference=0.5, overlap_percentage=0.1,
                 accept_threshold=0.7, minimum_factor=1.0):
        self.factor_difference = factor_difference
        self.overlap_percentage = overlap_percentage
        self.accept_threshold = accept_threshold
        self.minimum_factor = minimum_factor

class TrainingOptions():
    '''Training options object to store testing proportion configuration and equalisation option.
    Args:
        testing_proportion (float): Defaults to 0.15. Proportion of dataset used as testing data.
        equalise (bool): If the datasets should be of equal size.
    '''
    def __init__(self, testing_proportion=0.15, equalise=True):
        self.testing_proportion = testing_proportion
        self.equalise = equalise

class Model():
    '''Model options object to store SVM model, accuracy and configuration in one object.
    Args:
        svm_model (LinearSVM): sklearn SVM model.
        accuracy (float): Accuracy of model in percentage form.
        hog_options (HOGOptions): Configuration of HOG algorithm used to train SVM.
    '''
    def __init__(self, svm_model, accuracy, hog_options):
        self.svm_model = svm_model
        self.accuracy = accuracy
        self.hog_options = hog_options

def load_image(path, as_gray=True):
    '''Load image
    Args:
        path (string): Filepath of image to load.
    Returns:
        numpy.array: Uses skimage's built-in method to load image.
    '''
    return imread(path, as_gray=as_gray)

def _sliding_window(image, model, scale, hog_options=HOGOptions(), detection_options=DetectionOptions()):
    '''A sliding window worker method to find faces using image pyramid scales.
    Args:
        image (list): A black and white list represented image to be scanned.
        model (LinearSVM): An sklearn linear svm model with proba trained with hog data.
        hog_options (HOGOptions): Defaults to HOGOptions().
        detection_options (DetectionOptions): Defaults to DetectionOptions().
        scale (double): Scale of the image.
    Returns:
        found_faces (list): All suspected faces.
    '''
    found_faces = list()
    # Rescale the image to form an image pyramid.
    rescaled_image = rescale(image, 1/scale, mode='reflect')
    # Calculate overlap pixel size from dimension average.
    overlap_amount = int(sum(rescaled_image.shape) * 0.5 * detection_options.overlap_percentage)
    # Go through the vertical and horizontal pixels to form a sliding window.
    for row_start in range(0, rescaled_image.shape[0] - hog_options.window_size[0], overlap_amount):
        for column_start in range(0, rescaled_image.shape[1] - hog_options.window_size[1],
                                  overlap_amount):
            row_end = row_start + hog_options.window_size[0]
            column_end = column_start + hog_options.window_size[1]
            # Crop the desired window from the image.
            window = rescaled_image[row_start:row_end, column_start:column_end]
            # Calculate the Histogram of Orientate Gradients for the desired window.
            window_hog = hog(window, orientations=hog_options.orientations,
                             pixels_per_cell=hog_options.pixels_per_cell,
                             cells_per_block=hog_options.cells_per_block)
            # Calculate the probability of the window containing a face.
            model_probability = model.predict_proba([window_hog])[:, 1][0]
            if model_probability >= detection_options.accept_threshold:
                # Scale the window coordinates to the full size image
                found_faces.append(Face((int(row_start*scale), int(column_start*scale)), (int(row_end*scale),
                                   int(column_end*scale)), model_probability))
    return found_faces

def find_all_face_boxes(image, complete_model, detection_options=DetectionOptions()):
    '''A function to create _sliding_window() processes to detect faces.
    Args:
        image (list): An list represented image to scan.
        complete_model (Model): An object containing a sklearn LinearSVM model with proba and the
                                HOG configuration it was trained with.
        detection_options (DetectionOptions): Defaults to DetectionOptions().
    Returns:
        possible_faces (list): A list containing detected faces in the form of Face objects.
    '''
    model = complete_model.svm_model
    hog_options = complete_model.hog_options
    # Raise an IndexError if file too small. 
    if image.shape[0] < hog_options.window_size[0] or image.shape[1] < hog_options.window_size[1]:
        print('Image is too small for set window size')
        raise IndexError
    # Calculate the maximum factor that the window can be multiplied by according to the size of the image.
    if image.shape[0] < image.shape[1]:
        maximum_factor = image.shape[0] / hog_options.window_size[0]
    else:
        maximum_factor = image.shape[1] / hog_options.window_size[1]
    image = rgb2gray(image)
    # Calculate values to factor by for image pyramid.
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
    '''Generate Histogram of Oriented Gradients features from given image.
    Args:
        image (numpy.array): Image to calculate features from as a numpy array.
        hog_options (HOGOptions, optional): Defaults to HOGOptions(). Configuration for HOG algorithm.
    Returns:
        hog_image (numpy.array): Features of HOG data extracted from image.
    '''
    # Check if image is correctly sized, if not resize. This may cause images to be distorted and is not prefered.
    if image.shape != hog_options.window_size:
        print('Resizing this could potentially lead to bad data')
        image = resize(image,hog_options.window_size)
    # Calculate the Histogram of Orientate Gradients for the desired window with defaults.
    hog_image = hog(image)
    return hog_image

def generate_hog_data_from_dir(folder_path, hog_options=HOGOptions(), limit=None):
    '''Generate Histogram of Oriented Gradient features from given directory.
    Args:
        folder_path (string): Folder path of data.
        hog_options (HOGOptions, optional): Defaults to HOGOptions(). Configuration for
                                            HOG algorithm.
        limit (integer, optional): Defaults to None. Limit to amount of data to be imported.
    Returns:
        hog_data (list): HOG data for each image in directory.
    '''
    # Output all images in given directory
    images = listdir(path=folder_path)
    # Remove images that go over the limit
    if limit:
        images = images[:limit]
    # Iterate over each given image
    hog_data = list()
    for image_name in images:
        try:
            # Read image as gray and join name to filepath
            image = load_image(join(folder_path, image_name), as_gray=True)
            hog_image = generate_hog_data(image, hog_options=hog_options)
            hog_data.append(hog_image)
        except FileNotFoundError:
            # If file is not found then throw error message
            print('Failed to load '+image_name)
    return hog_data

def _calculate_equalise(x, y):
    '''Calculate the size the datasets should be cropped to.
    Args:
        x (integer): First dataset size.
        y (integer): Second dataset size.
    Returns:
        integer: Size datasets should be cropped to.
    '''
    return min(x, y)

def split_training_data(x, y, test_percentage=0.2, equalise=True):
    '''This splits data into training and testing sets using a given percentage.
    Args:
        x (list): The first set of data.
        y (list): The second set of data.
        test_percentage (float, optional): Defaults to 0.2. Percentage to be used as testing data.
        equalise (bool, optional): Defaults to True. If datasets should be equal sizes.
    Returns:
        x (list): First dataset for training.
        y (list): Second dataset for training.
        x_test (list): First dataset for testing.
        y_test (list): Second dataset for testing.
    '''
    # If equalisation enabled and possible with data, equalise
    if equalise and len(x) != len(y):
        max_value = _calculate_equalise(len(x), len(y))
        x = x[:max_value]
        y = y[:max_value]
    # Calculate testing sizes for the two datasets.
    x_test_size = int(len(x)*test_percentage)
    y_test_size = int(len(y)*test_percentage)
    # Sort the datasets into training and testing.
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
        classifier (LinearSVM): just the completed SVM model.
        score (float): the accuracy of the model.
    """
    x = list()
    y = list()
    x_test = list()
    y_test = list()
    # Assign positive data with label of 1 to mark it is positive
    for data in positive_train:
        x.append(data)
        y.append(1)
    # Assign negative data with label of 0 to mark it is negative
    for data in negative_train:
        x.append(data)
        y.append(0)
    # Assign test data similarly
    for data in positive_test:
        x_test.append(data)
        y_test.append(1)
    for data in negative_test:
        x_test.append(data)
        y_test.append(0)
    # Briefly calculate best values for training
    parameter_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(LinearSVC(), parameter_grid)
    grid_search.fit(x, y)
    # Calculate SVM using all data given
    svm = grid_search.best_estimator_
    classifier = CalibratedClassifierCV(svm)
    classifier.fit(x, y)
    # Multiply to a percentage and round to 2 decimal places
    score = round(classifier.score(x_test, y_test) * 100)
    return classifier, score

def train(positive_path, negative_path, hog_options=HOGOptions(), train_options=TrainingOptions()):
    '''Train LinearSVM given just positive and negative data filepaths and configuration info.
    Args:
        positive_path (string): Filepath of the folder containing the positive dataset.
        negative_path (string): Filepath of the folder containing the negative dataset.
        hog_options (HOGOptions, optional): Defaults to HOGOptions(). Configuration for the HOG
                                            feature extraction algorithm.
        train_options (TrainingOptions, optional): Defaults to TrainingOptions(). Configuration
                                                   for training that's not related to HOG.
    Returns:
        model (Model): Model containing trained LinearSVM, HOG configuration and rounded score.
    '''
    # Generate positive and negative HOG features
    positive_hog = generate_hog_data_from_dir(positive_path, hog_options=hog_options)
    negative_hog = generate_hog_data_from_dir(negative_path, hog_options=hog_options)
    # Split training data for testing and training
    pos_train, neg_train, pos_test, neg_test = split_training_data(positive_hog, negative_hog,
        test_percentage=train_options.testing_proportion, equalise=train_options.equalise)
    # Train SVM
    svm_model, score = premade_train(pos_train, neg_train, pos_test, neg_test)
    # Contain SVM model and HOG configuration inside a Model object
    model = Model(svm_model, round(score), hog_options)
    return model
