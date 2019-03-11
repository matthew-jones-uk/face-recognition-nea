import numpy as np

class HOGOptions():
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 nbins=9, window_size=(64, 64)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.nbins = nbins
        self.window_size = window_size

def calc_gradient_magnitude(image):
    gradient = np.zeros(image.shape)
    magnitude = np.zeros(image.shape)
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            if h > 0 and w > 0 and h < image.shape[0] and w < image.shape[1]:
                # simplified calculation of gradient in x and y direction using kernel [-1, 0, 1]
                gy = image[h+1][w] - image[h-1][w]
                gx = image[h][w+1] - image[h][w-1]
                # gradient direction calculation and conversion from radians to degrees
                gradient[h][w] = np.arctan2(gy, gx) * (180 / np.pi)
                # remove any values below 0 to give range 0-180
                if gradient[h][w] < 0:
                    gradient[h][w] += 180
                # simple magnitude calculation 
                magnitude[h][w] = np.sqrt(gy**2, gx**2)
    return gradient, magnitude


