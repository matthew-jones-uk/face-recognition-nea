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
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if y > 0 and x > 0 and y < image.shape[0] and x < image.shape[1]:
                # simplified calculation of gradient in x and y direction using kernel [-1, 0, 1]
                gy = image[y+1][x] - image[y-1][x]
                gx = image[y][x+1] - image[y][x-1]
                # gradient direction calculation and conversion from radians to degrees
                gradient[y][x] = np.arctan2(gy, gx) * (180 / np.pi)
                # remove any values below 0 to give range 0-180
                if gradient[y][x] < 0:
                    gradient[y][x] += 180
                # simple magnitude calculation 
                magnitude[y][x] = np.sqrt(gy**2, gx**2)
    return gradient, magnitude

def create_histogram(gradient, magnitude, options=HOGOptions()):
    # calculate number of cells in x and y directions
    n_cells_y = gradient.shape[0] // options.pixels_per_cell[0]
    n_cells_x = gradient.shape[1] // options.pixels_per_cell[1]
    # resize image so that any pixels that don't fit into cells are discarded
    image_size_y = n_cells_y * options.pixels_per_cell[0]
    image_size_x = n_cells_x * options.pixels_per_cell[1]
    gradient = gradient[:image_size_y, :image_size_x]
    magnitude = magnitude[:image_size_y, image_size_x]
    # create cells
    cells = np.zeros((n_cells_y, n_cells_x, options.nbins))
    bin_size = 180 // options.nbins
    for y in range(gradient.shape[0]):
        for x in range(gradient.shape[1]):
            # calculates which cell and orientation bin each pixel belongs to and adds its magnitude
            cells[y // n_cells_y][x // n_cells_x][magnitude // bin_size] += magnitude[y][x]
    
