import tensorflow as tf 
from MathVisual.src.occ.dataset import symbols_list

# Create RESNET-50 for transfer learning
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