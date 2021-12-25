import os
import numpy as np 

symbols_list = ['zero','one','two','three','four','five','six','seven','eight','nine','minus','plus','equal','div','decimal','times']
from photomath.src.occ.dataset import DatasetLoader
path_dataset = "/home/{0}/Downloads/archive/".format(os.getlogin())
dataset_occ = DatasetLoader(path_dataset)
X_train,X_test = dataset_occ.create_train_test_data()
y_train, y_test = dataset_occ.create_labels()

import tensorflow as tf 
input_tensor = tf.keras.Input(shape=(100,100,3))
efnet = tf.keras.applications.ResNet50(weights='imagenet',
                                             include_top = False, 
                                             input_tensor = input_tensor)
# Now that we apply global max pooling.
gap = tf.keras.layers.GlobalMaxPooling2D()(efnet.output)
# Finally, we add a classification layer.
output = tf.keras.layers.Dense(symbols_list.__len__(), activation='softmax', use_bias=True)(gap)
# bind all
func_model = tf.keras.Model(efnet.input, output)
func_model.compile(
          loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())
# Loads the weights
model_path = "/home/{0}/Furkan/coding/repos_me/PhotoMath/checkpoints/models_resnet_acc.h5".format(os.getlogin())
func_model.load_weights(model_path)
# Re-evaluate the model
loss, acc = func_model.evaluate(X_train, y_train, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

