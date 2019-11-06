import numpy as np
import matplotlib.pyplot as plot
import matplotlib.image as img


def load_mnist_images_labels(images_path: str, labels_path: str):
    """
    Function loads mnist files and converts it to numpy matrices.
    :param images_path: path to file that contains images from mnist page
    :param labels_path: path to file that contains labels from mnist page
    :return: images, labels as numpy matrices
    """

    # firstly load image and label file
    try:
        data_images = np.fromfile(images_path, 'ubyte')
        data_labels = np.fromfile(labels_path, 'ubyte')
    except OSError as error:
        print("Exception occured while loading files ",  error)
        raise error
    else:
        # Header contains 4x4bytes information: magic_number, images_count, width, height
        images_header_size = 4 * 4
        label_header_size = 2 * 4
        int_type = np.dtype('int32').newbyteorder('>')
        magic_number, images_count, width, height = np.frombuffer(data_images[:images_header_size], int_type)
        images = data_images[images_header_size:].astype('float32').reshape([images_count, width * height, 1])
        labels = data_labels[label_header_size:]
        #labels = labels_matrix_transf(labels)

    return images, labels

def labels_matrix_transf(labels):
    return list(map(__create_matrix_, labels))

def __create_matrix_(number):
    matrix = np.zeros([10,1], dtype='float32')
    matrix[number] = 1.0
    return matrix

if __name__ == "__main__":
    images, labels = load_mnist_images_labels("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte")
    plot.imshow(images[10], 'gray')
    plot.axis('off')
    plot.title(labels[10])
    print(labels_matrix_transf(labels)[10])
    print("hello")