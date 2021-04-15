import logging
import os
import time

import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.models import Sequential

from sklearn.model_selection import train_test_split


def load_data(training_file):
	df = pd.read_csv(training_file)
	features = df.iloc[:, 1:]
	target = df.iloc[:, 0]
	target_map = {}
	for idx, value in enumerate(target.unique()):
		target_map[value] = idx

	return features, target, target_map


def train_save_model(training_file, output_dir, epochs=3):
	features, target, target_map = load_data(training_file)

	train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.1)
	train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.1)
	model = Sequential()
	model.add(LSTM(8, input_dim=6, name="input"))
	model.add(Dropout(0.1))
	model.add(LSTM(8))
	model.add(Dropout(0.1))
	model.add(Dense(1, activation='relu', name="output"))

	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	logging.info("Model has been compiled")

	model.fit(train_X, train_Y, epochs=epochs, validation_data=(val_X, val_Y))

	# Save Model
	# Use TF to save the graph model instead of Keras save model to load it in Golang
	save_dir_path = os.path.join(output_dir, "model_"+str(int(time.time())))
	os.mkdir(save_dir_path)
	tf.saved_model.save(model, save_dir_path)
	# model.save_pretrained(save_dir_path, saved_model=True)
	logging.info("Model Saved at: {}".format(save_dir_path))

	# Evaluate
	test_loss, test_acc = model.evaluate(test_X, test_Y)
	logging.info("Test loss: {}, Test Accuracy: {}".format(test_loss, test_acc))
