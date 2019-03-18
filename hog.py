import numpy as np

class HOGOptions():
    '''Object to configure HOG algorithm.
    Args:
        nbins (int, optional): Defaults to 9. Number of orientation bins for histogram.
        pixels_per_cell (tuple, optional): Defaults to (8, 8). Should be factors of window size.
        cells_per_block (tuple, optional): Defaults to (2, 2). Cell number should be a factor.
        window_size (tuple, optional): Defaults to (64, 64). Size of image window in pixels.
    '''
    def __init__(self, nbins=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                 window_size=(64, 64)):
        self.nbins = nbins
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.window_size = window_size

def calc_gradient_direction_magnitude(image):
    '''Calculate gradient direction and gradient magnitude for each pixel.
    Args:
        image (numpy.array): Numpy array of a black and white image.
    Returns:
        gradient (numpy.array): Numpy array of gradient direction.
        magnitude (numpy.array): Numpy array of gradient magnitude.
    '''

    gradient = np.zeros(image.shape)
    magnitude = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if y > 0 and x > 0 and y < image.shape[0]-1 and x < image.shape[1]-1:
                # simplified calculation of gradient in x and y direction using kernel [-1, 0, 1]
                gy = image[y+1][x] - image[y-1][x]
                gx = image[y][x+1] - image[y][x-1]
                # gradient direction calculation and conversion from radians to degrees
                gradient[y][x] = np.arctan2(gy, gx) * (180 / np.pi)
                # remove any values below 0 to give range 0-180
                if gradient[y][x] < 0:
                    gradient[y][x] += 180
                # simple magnitude calculation
                magnitude[y][x] = np.sqrt(gy**2 + gx**2)
    return gradient, magnitude

def create_histogram(gradient, magnitude, options=HOGOptions()):
    '''Create final histogram from gradients and magnitudes.
    Args:
        gradient (numpy.array): Numpy array of gradient directions for each pixel.
        magnitude (numpy.array): Numpy array of gradient magnitudes for each pixel.
        options (HOGOptions, optional): Defaults to HOGOptions(). HOG algorithm configuration.
    Returns:
        blocks (list): 2D list of each block and corresponding histogram.
    '''
    # calculate number of cells in x and y directions
    n_cells_y = gradient.shape[0] // options.pixels_per_cell[0]
    n_cells_x = gradient.shape[1] // options.pixels_per_cell[1]
    # resize image so that any pixels that don't fit into cells are discarded
    image_size_y = n_cells_y * options.pixels_per_cell[0]
    image_size_x = n_cells_x * options.pixels_per_cell[1]
    gradient = gradient[:image_size_y-1, :image_size_x-1]
    magnitude = magnitude[:image_size_y-1, :image_size_x-1]
    # create cells
    cells = np.zeros((n_cells_y, n_cells_x, options.nbins))
    bin_size = 180 // options.nbins
    for y in range(gradient.shape[0]):
        for x in range(gradient.shape[1]):
            # calculate which orientation bin the pixel belongs to
            bin_number = int(gradient[y][x] // bin_size)
            # when gradient is 180 the bin size should fit into the previous range, not a new one
            if bin_number == options.nbins:
                bin_number -= 1
            # calculates which cell and orientation bin each pixel belongs to and adds its magnitude
            cells[y // n_cells_y][x // n_cells_x][bin_number] += magnitude[y][x]
    # calculate cell overlap amount
    cell_overlap_y = options.cells_per_block[0] // 2
    cell_overlap_x = options.cells_per_block[1] // 2
    # resize cells and discarded any that don't fit into a block
    remainder_y = cells.shape[0] // cell_overlap_y
    remainder_x = cells.shape[1] // cell_overlap_x
    cells = cells[:remainder_y, :remainder_x, :]
    # create blocks with overlap 50%
    blocks = list()
    y = 0
    while y < cells.shape[0]-1:
        x = 0
        while x < cells.shape[1]-1:
            # select cells that make up block
            block = cells[y:y+options.cells_per_block[0], x:x+options.cells_per_block[1], :]
            # combine into a single block
            block = np.sum(block, axis=(0, 1))
            # normalise using l2-norm
            normalised_block = block / np.sqrt(np.sum(block) + 1e-7)
            blocks.append(normalised_block.flatten())
            x += cell_overlap_x
        y += cell_overlap_y
    return blocks

def hog(image, options=HOGOptions()):
    '''Calculate Histograms of Oriented Gradients for image
    Args:
        image (numpy.array): Numpy array of black and white image.
        options (HOGOptions, optional): Defaults to HOGOptions(). HOG algorithm configuration.
    Returns:
        histogram (list): 2D list of each block and corresponding histogram.
    '''
    gradient, magnitude = calc_gradient_direction_magnitude(image)
    histogram = create_histogram(gradient, magnitude, options=options)
    return histogram