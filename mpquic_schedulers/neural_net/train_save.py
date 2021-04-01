import logging
import os
import time

import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from sklearn.model_selection import train_test_split


# path_id,cwnd_1,cwnd_2,in_flight_1,in_flight_2,rtt_1,rtt_2,avg_rtt_1,avg_rtt_2
def load_data(training_file):
	df = pd.read_csv(training_file)
	if len(df) <= 1:
		return [], [], [], True
	features = df.iloc[:, 1:]
	target = df.iloc[:, 0]
	target_map = {}
	for idx, value in enumerate(target.unique()):
		target_map[value] = idx

	return features, target, target_map, False


def train_save_model(training_file, epochs, output_dir):
	features, target, target_map, should_skip = load_data(training_file)
	if should_skip:
		logging.info("Skipping Model training, not enough data")
		return
	train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.1)
	train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.1)
	model = Sequential()
	model.add(Dense(12, input_shape=(8,), activation='linear', name="input"))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid', name="output"))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
