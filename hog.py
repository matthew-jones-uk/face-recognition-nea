import numpy as np

class HOGOptions():
    '''Object to configure HOG algorithm.
    Args:
        nbins (int, optional): Defaults to 9. Number of orientation bins for histogram.
        pixels_per_cell (tuple, optional): Defaults to (8, 8). Should be factors of window size.
        cells_per_block (tuple, optional): Defaults to (2, 2). Cell number should be a factor.
        window_size (tuple, optional): Defaults to (64, 64). Size of image window in pixels.
    '''
    def __init__(self, nbins=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), 
                 window_size=(64, 64)):
        self.nbins = nbins
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.window_size = window_size

def _calculate_gradients(image):
    '''Calculate gradient direction and gradient magnitude for each pixel.
    Args:
        image (numpy.array): Numpy array of a black and white image.
    Returns:
        gy (numpy.array): Numpy array of gradients in y direction.
        gx (numpy.array): Numpy array of gradients in x direction.
    '''
    # Create numpy arrays for gradient in x and y directions
    gy = np.empty(image.shape)
    gx = np.empty(image.shape)
    # Set first and last index of both directions to zero.
    gy[0, :] = 0
    gy[-1, :] = 0
    gx[:, 0] = 0
    gx[:, -1] = 0
    # Simplified calculation of gradient in x and y directions using kernel [-1, 0, 1]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    return gy, gx

def _calculate_cell(magnitude, orientation, nbins):
    if magnitude.shape != orientation.shape:
        raise ValueError
    # Create a numpy array of size number of bins
    bins = np.zeros(nbins)
    # Calculate the size of each bin
    bin_size = 180/nbins
    # Iterate through each bin and pixel and decide if it should be considered.
    for bin_index in range(nbins):
        range_start = bin_index*bin_size
        range_end = (bin_index+1)*bin_size
        for y in range(orientation.shape[0]):
            for x in range(orientation.shape[1]):
                if orientation[y, x] >= range_start and orientation[y, x] < range_end:
                    bins[bin_index] += magnitude[y, x]
    return bins

def _normalise_block(block, eps = 1e-5):
    # Calculate L2-Hys for block normalisation.
    value = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    value = np.minimum(value, 0.2)
    value = value / np.sqrt(np.sum(value ** 2) + eps ** 2)
    return value

def hog(image, options=HOGOptions()):
    '''Calculate Histograms of Oriented Gradients for image
    Args:
        image (numpy.array): Numpy array of black and white image.
        options (HOGOptions, optional): Defaults to HOGOptions(). HOG algorithm configuration.
    Returns:
        histogram (list): 2D list of each block and corresponding histogram.
    '''
    gy, gx = _calculate_gradients(image)
    # Calculate number of cells in y and x axis.
    n_cells_y = int(image.shape[0] // options.pixels_per_cell[0])
    n_cells_x = int(image.shape[1] // options.pixels_per_cell[1])
    # Calculate histogram for cells
    cell_histogram = np.zeros((n_cells_y, n_cells_x, options.nbins))
    magnitude = np.hypot(gy, gx)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180
    # Calculate cells
    for y_cell in range(n_cells_y):
        y_start = y_cell * options.pixels_per_cell[0]
        y_end = y_start + options.pixels_per_cell[0]
        for x_cell in range(n_cells_x):
            x_start = x_cell * options.pixels_per_cell[1]
            x_end = x_start + options.pixels_per_cell[1]
            cell = _calculate_cell(magnitude[y_start:y_end, x_start:x_end],
                                  orientation[y_start:y_end, x_start:x_end],
                                  options.nbins)
            cell_histogram[y_cell, x_cell, :] = cell
    # Calculate number of blocks in y and x axis
    n_blocks_y = int(n_cells_y // options.cells_per_block[0])
    n_blocks_x = int(n_cells_x // options.cells_per_block[1])
    # Sort cells into blocks
    blocks = np.zeros((n_blocks_y, n_blocks_x, options.cells_per_block[0],
                       options.cells_per_block[1], options.nbins))
    for y_block in range(n_blocks_y):
        for x_block in range(n_blocks_x):
            block = cell_histogram[y_block:y_block + options.cells_per_block[0],
                                   x_block:x_block + options.cells_per_block[1], :]
            blocks[y_block, x_block, :] = _normalise_block(block)
    # Flatten
    blocks = blocks.ravel()
    return blocks
