import tensorflow as tf
import matplotlib.pyplot as plot
from numpy import argmax

class NeuralNetwork:
    def __init__(self, network_layers: (int)):
        network_struct = [tf.keras.layers.Flatten(input_shape=network_layers[0])]
        for layer in network_layers[1:]:
            network_struct.append(tf.keras.layers.Dense(layer, activation='relu'))

        self.__neural_network = tf.keras.Sequential(network_struct)
        self.__neural_network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def teach(self, train_data, train_labels, epoch: int):
        self.__neural_network.fit(train_data, train_labels, epoch=epoch)

    def test(self, test_data, test_labels):
        return self.__neural_network.evaluate(test_data, test_labels, verbose=2)

    def guess(self, image):
        prediction = self.__neural_network.predict(image)
        return prediction

    def save(self, filePath: str):
        """
        Save neural network to specified file with .h2 extension
        :param filePath: file name/path without extension(added in function)
        """
        self.__neural_network.save(str + ".h2")

    @classmethod
    def load(cls, filePath: str):
        # creating dope object
        loaded_nn = cls( ( (1, 1), 1) )
        model = tf.keras.models.load_model(filePath)
        loaded_nn.__neural_network = model
        return loaded_nn



if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    print(train_images.shape)

    # normalizing images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #plot.figure()
    # for i in range(10):
    #     plot.subplot(5, 2, i+1)
    #     plot.xticks([])
    #     plot.yticks([])
    #     plot.imshow(train_images[i], cmap=plot.cm.binary)
    #     plot.xlabel(train_labels[i])
    # plot.show()

    neural_network = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), \
                                          tf.keras.layers.Dense(128, activation='relu'), \
                                         tf.keras.layers.Dense(32, activation='relu'), \
                                         tf.keras.layers.Dense(10, activation='softmax')])

    neural_network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    #teaching neural network
    neural_network.fit(train_images, train_labels, epochs=10)
    neural_network.save('image_recog.h5')

    test_loss, test_acc = neural_network.evaluate(test_images, test_labels, verbose=2)

    print("\nTest accuracy:", test_acc)