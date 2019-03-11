import numpy as np

class HOGOptions():
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, window_size=(64, 64)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.nbins = nbins
        self.window_size = window_size

def gradient(image):
    size_y, size_x = image.shape
    # calculate gradient in the x direction
    gx = np.zeros((size_y-2, size_x-2))
    gx[:, :] = image[1:-2, 2:] - image[1:-1, :-2]
    # calculate gradient in the y direction
    gy = np.zeros((size_y-2, size_x-2))
    gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]
    return gx, gy

def orientation_magnitude(gx, gy):
    # calculate orientation and convert it from radians to degrees
    orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 360
    magnitude = np.sqrt(gx**2, gy**2)
    return orientation, magnitude
def calculate_histogram(magnitude, orientation, options=HOGOptions()):
    ...
