import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python import keras


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y


data, info = tfds.load('mnist', data_dir='data', with_info=True, as_supervised=True)
print(info)

train_dataset = data['train'].map(preprocess).batch(256)
test_dataset = data['test'].map(preprocess).batch(256)

model = keras.Sequential([
    keras.layers.Reshape((28, 28), input_shape=(28, 28, 1)),
    keras.layers.CuDNNLSTM(128),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()

opt = keras.optimizers.SGD(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=300,
                    validation_data=test_dataset)
