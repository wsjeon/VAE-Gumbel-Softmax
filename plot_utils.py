import numpy as np
from matplotlib import pyplot as plt


def make_squares(images, nr_images_per_side):
    images_to_plot = np.concatenate(
        [np.concatenate([images[j*nr_images_per_side+i].reshape((28,28)) for i in range(0,nr_images_per_side)],
                        axis=1)
         for j in range(0,nr_images_per_side)],
        axis=0)
    return images_to_plot


def plot_squares(originals, reconstructs, generations, nr_images_per_side):
    originals_square = make_squares(originals, nr_images_per_side)
    plt.imsave('./results/original.png', originals_square, cmap='viridis')
    reconstructs_square = make_squares(reconstructs, nr_images_per_side)
    plt.imsave('./results/recons.png', reconstructs_square, cmap='viridis')
    combined = np.concatenate([originals_square, reconstructs_square], axis=1)
    plt.imsave('./results/combined.png', combined, cmap='viridis')
    generations_square = make_squares(generations, nr_images_per_side)
    plt.imsave('./results/generations.png', generations_square, cmap='viridis')
