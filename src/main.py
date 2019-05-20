import argparse

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python import keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    return parser.parse_args()


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y


def main():
    args = parse_args()
    print(vars(args))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    data, info = tfds.load('mnist', data_dir='data', with_info=True, as_supervised=True)
    print(info)

    train_dataset = data['train'].map(preprocess).batch(args.batch_size)
    test_dataset = data['test'].map(preprocess).batch(args.batch_size)

    model = keras.Sequential([
        keras.layers.Reshape((28, 28), input_shape=(28, 28, 1)),
        keras.layers.CuDNNLSTM(128),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()

    opt = keras.optimizers.SGD(lr=args.lr)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(train_dataset,
              epochs=args.epochs,
              validation_data=test_dataset)


if __name__ == '__main__':
    main()
