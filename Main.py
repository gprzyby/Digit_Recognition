from matplotlib.patches import Rectangle

from MnistLoader import load_mnist_images_labels as loadData
import numpy as np
import matplotlib.pyplot as plot

if __name__ == "__main__":
    images, train_labels = loadData("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte")
    _, test_labels = loadData("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte")
    plot.xticks(range(10))
    plot.xlabel("Number")
    plot.ylabel("Amount in data")
    train_hist = plot.hist(train_labels, bins=10, label="Train histogram", ec='black', color='blue')
    test_hist = plot.hist(test_labels, bins=10, label="Test histogram", ec='black', color='red')

    handles = [Rectangle((0, 0), 1., 1., color=c) for c in ["blue", "red"]]
    label = ["Train histogram", "Test histogram"]
    plot.legend(handles, label)
    plot.show()
