import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import MNIST data
mnist = input_data.read_data_sets('data', one_hot=True)

# training parameters
learning_rate = 0.001
batch_size = 128
num_epochs = 300
display_step = 40

# network parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

# tf graph input
X = tf.placeholder(tf.float32, [None, timesteps, num_input])  # (batch_size, timesteps, n_input)
Y = tf.placeholder(tf.int64, [None, num_classes])  # (batch_size, n_classes)

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_hidden)
outputs, state = tf.nn.static_rnn(cell, inputs=tf.unstack(X, timesteps, 1), dtype=tf.float32)
logits = tf.contrib.layers.linear(inputs=outputs[-1], num_outputs=num_classes)

# loss
loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_fn = optimizer.minimize(loss_fn)

# compute accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)), tf.float32))


# initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

try:
    # start training
    with tf.Session() as sess:
        # run the initializer
        sess.run(init)

        for epoch in range(num_epochs):
            for iteration in range(int(mnist.train.num_examples / batch_size)):
                train_x, train_y = mnist.train.next_batch(batch_size)
                # reshape data to get 28 seq of 28 elements
                train_x = train_x.reshape((batch_size, timesteps, num_input))
                # run backprop
                sess.run(train_fn, feed_dict={X: train_x, Y: train_y})

                if iteration%display_step == 0:
                    loss = sess.run(loss_fn, feed_dict={X: train_x, Y: train_y})
                    print('Step: {}, loss: {:.6f}'.format(iteration, loss))

            # evaluate on validation data
            val_acc = 0
            val_iters = int(mnist.validation.num_examples / batch_size)
            for _ in range(val_iters):
                val_x, val_y = mnist.validation.next_batch(batch_size)
                val_x = val_x.reshape((batch_size, timesteps, num_input))
                val_iter_acc = sess.run(accuracy, feed_dict={X: val_x, Y: val_y})
                val_acc += val_iter_acc
            val_acc /= val_iters

            # evaluate on test data
            test_acc = 0
            test_iters = int(mnist.test.num_examples / batch_size)
            for _ in range(test_iters):
                test_x, test_y = mnist.test.next_batch(batch_size)
                test_x = test_x.reshape((batch_size, timesteps, num_input))
                test_iter_acc = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
                test_acc += test_iter_acc
            test_acc /= test_iters

            print('Epoch {}/{}, '
                  'validation accuracy: {:.2f}%, '
                  'test accuracy: {:.2f}%'.format(epoch + 1,
                                                  num_epochs,
                                                  100. * val_acc,
                                                  100. * test_acc))

    print("Optimization finished!")
except KeyboardInterrupt:
    pass
