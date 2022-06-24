# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    bbc_sentences = bbc['text']
    bbc_labels = bbc['category']
    x_train, x_val, y_train, y_val = train_test_split(bbc_sentences, bbc_labels, train_size=training_portion,
                                                      shuffle=False)

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)

    sequences = tokenizer.texts_to_sequences(x_train)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    val_sequences = tokenizer.texts_to_sequences(x_val)
    val_padded = pad_sequences(val_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    encoder = LabelEncoder()
    encoder.fit(y_train)

    y_train_final = encoder.transform(y_train)
    y_val_final = encoder.transform(y_val)

    # val_tokenizer = Tokenizer(oov_token=oov_tok)
    # val_tokenizer.fit_on_texts(y_train)

    # y_train_final = val_tokenizer.texts_to_sequences(y_train)
    # y_val_final = val_tokenizer.texts_to_sequences(y_val)
    # y_train_final = np.array(y_train_final)
    # y_val_final = np.array( y_val_final)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(padded,
              y_train_final,
              epochs=50,
              validation_data=(val_padded, y_val_final))

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.


if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
