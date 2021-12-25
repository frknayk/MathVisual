import os
import tensorflow as tf 
from photomath.src.occ.dataset import DatasetLoader, symbols_list
import matplotlib.pyplot as plt

path_dataset = "/home/{0}/Downloads/archive/".format(os.getlogin())
dataset_occ = DatasetLoader(path_dataset)
X_train,X_test = dataset_occ.create_train_test_data()
y_train, y_test = dataset_occ.create_labels()

input_tensor = tf.keras.Input(shape=(100,100,3))
efnet = tf.keras.applications.ResNet50(weights='imagenet',
                                             include_top = False, 
                                             input_tensor = input_tensor)
# Now that we apply global max pooling.
gap = tf.keras.layers.GlobalMaxPooling2D()(efnet.output)
# Finally, we add a classification layer.
output = tf.keras.layers.Dense(symbols_list.__len__(), activation='softmax', use_bias=True)(gap)
model = tf.keras.Model(efnet.input, output)
model.compile(
          loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())

# Train
history = model.fit(
    X_train, 
    y_train, 
    batch_size=50, 
    epochs=200, 
    validation_split=0.2)

# Save model
model.save("checkpoints/models_resnet_acc.h5")

# plot training curves
plt.figure(1)
plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['categorical_accuracy'],label="acc_categorical")
plt.title('Training Curve')
plt.legend()
plt.xlabel('epoch')
plt.show()

