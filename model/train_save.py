import logging
import os
import time

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Softmax
from tensorflow.python.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#
# def norm_data(training_file):
#     features, target, target_map = load_data(training_file)
#     input_scalar = preprocessing.MinMaxScaler()
#     scaled = input_scalar.fit_transform(target)
#     return input_scalar.fit_transform(features), scaled


def load_data(training_file):
    df = pd.read_csv(training_file)
    input_scalar = preprocessing.MinMaxScaler()
    features = pd.DataFrame(input_scalar.fit_transform(df.iloc[:, 2:]))
    target_scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
    target = pd.DataFrame(target_scalar.fit_transform(df.iloc[:, :2]))
    target = target.round(3)
    return features, target


def train_save_model(training_file, output_dir, epochs=3):
    features, target = load_data(training_file)
    target = np.array(target)
    print(np.shape(features), np.shape(target), target)
    train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.1)
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.1)
    train_X = np.reshape(np.array(train_X), (np.shape(train_X)[0], np.shape(train_X)[1], 1))
    val_X = np.reshape(np.array(val_X), (np.shape(val_X)[0], np.shape(val_X)[1], 1))
    test_X = np.reshape(np.array(test_X), (np.shape(test_X)[0], np.shape(test_X)[1], 1))
    
    model = Sequential()
    model.add(LSTM(10, input_shape=(6, 1), name="input", return_sequences=True, recurrent_activation="tanh",
                   kernel_initializer='normal'))
    model.add(LSTM(100, recurrent_activation="tanh", activation="tanh", kernel_initializer='normal'))
    model.add(Dense(2, kernel_initializer='normal'))
    model.add(Softmax(name="output"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8, epsilon=1e-7)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    
    logging.info("Model has been compiled")
    model.fit(train_X, train_Y, epochs=epochs, validation_data=(val_X, val_Y))
    
    # Save Model
    # Use TF to save the graph model instead of Keras save model to load it in Golang
    save_dir_path = os.path.join(output_dir, "model_" + str(int(time.time())))
    os.mkdir(save_dir_path)
    tf.saved_model.save(model, save_dir_path)
    # model.save_pretrained(save_dir_path, saved_model=True)
    logging.info("Model Saved at: {}".format(save_dir_path))
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_X, test_Y)
    logging.info("Test loss: {}, Test Accuracy: {}".format(test_loss, test_acc))
    preds = model.predict(test_X)
    print(preds)
    print(test_Y)
