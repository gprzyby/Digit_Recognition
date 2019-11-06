from MnistLoader import load_mnist_images_labels as loadData
import numpy as np
import matplotlib.pyplot as plot

if __name__ == "__main__":
    images, train_labels = loadData("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte")
    _, test_labels = loadData("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte")
    plot.hist(train_labels, bins=10, label="Train histogram")
    plot.hist(test_labels, bins=10, label="Test histogram")

    plot.show()
