# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


def solution_C2():
    mnist = tf.keras.datasets.mnist

    # NORMALIZE YOUR IMAGE HERE
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train = x_train / 255.
    x_val = x_val / 255.

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                                        tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(10, activation='softmax')])

    # COMPILE MODEL HERE
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    model.fit(x_train,
              y_train,
              validation_data=(x_val, y_val),
              epochs=20,
              verbose=1)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")
