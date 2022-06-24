# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # NORMALIZE YOUR IMAGE HERE
    (x_train, y_train), (x_val, y_val) = fashion_mnist.load_data()
    # x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_train = x_train / 255.0
    # x_val = x_val.reshape(len(x_val), 28, 28, 1)
    x_val = x_val / 255.0

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
                                        tf.keras.layers.MaxPooling2D(2, 2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(10, activation='softmax')])
    # COMPILE MODEL HERE
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    model.fit(x = x_train,
              y = y_train,
              validation_data = (x_val, y_val),
              epochs = 10,
              verbose = 1)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
