# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from photomath.src.occ.dataset import DatasetLoader, symbols_list
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Create the dataset loader object
import os
path_dataset = "/home/{0}/Downloads/archive/".format(os.getlogin())
dataset_loader = DatasetLoader(path_dataset)
X_train,X_test = dataset_loader.create_train_test_data()
y_train_labels, y_test_labels = dataset_loader.get_labels()

# Create labels
le = preprocessing.LabelEncoder()
le.fit(symbols_list)
y_train_temp = le.transform(y_train_labels)
y_test_temp = le.transform(y_test_labels)
y_train = keras.utils.to_categorical(y_train_temp, 16)
y_test = keras.utils.to_categorical(y_test_temp, 16)



model = Sequential()
# 1st layer and taking input in this of shape 100x100x3 ->  100 x 100 pixles and 3 channels
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
# maxpooling will take highest value from a filter of 2*2 shape
model.add(MaxPooling2D(pool_size=(2, 2)))
# it will prevent overfitting by making it hard for the model to idenify the images
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
# last layer predicts 16 labels
model.add(Dense(16, activation="softmax"))
# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)
history = model.fit(
            X_train,
            y_train,
            batch_size=128,
            epochs=3,
            validation_split=0.2,
            shuffle=True)


predictions = model.predict(X_train)
print("======= predictions ======")
print(predictions[0])
print(np.argmax(predictions[0]))
print(y_test_labels[0])

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions_prob_model = probability_model.predict(X_train)
print(predictions_prob_model[0])
print(np.argmax(predictions_prob_model[0]))
print(y_test_labels[0])