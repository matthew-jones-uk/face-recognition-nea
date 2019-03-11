import numpy as np

def gradient(image):
    size_y, size_x = image.shape
    gx = np.zeros((size_y-2, size_x-2))
    gx[:, :] = image[1:-2, 2:] - image[1:-1, :-2]
    gy = np.zeros((size_y-2, size_x-2))
    gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]
    return gx, gy

def orientation_magnitude(gx, gy):
    orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 360
    magnitude = np.sqrt(gx**2, gy**2)
    return orientation, magnitude

