import argparse
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python import keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
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
        keras.layers.Reshape((784, 1), input_shape=(28, 28, 1)),
        keras.layers.CuDNNGRU(128),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()

    opt = keras.optimizers.Adam(lr=args.lr, clipnorm=1.)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    model.fit(train_dataset,
              epochs=args.epochs,
              validation_data=test_dataset,
              callbacks=[tensorboard, early_stopping])

    scores = model.evaluate(test_dataset, verbose=0)
    for metric, score in zip(model.metrics_names, scores):
        print('{}: {}'.format(metric, score))


if __name__ == '__main__':
    main()
