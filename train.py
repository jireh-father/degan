import tensorflow as tf

batch_size = 2
input_size = 160
num_channel = 3
num_variation_factor = 3

input_ph = tf.placeholder(tf.float32, [batch_size, input_size, input_size, num_channel], "input_placeholder")
label_ph = tf.placeholder(tf.float32, [batch_size, num_variation_factor], "label_placeholder")
dropout_rate_ph = tf.placeholder(tf.float32, (), "dropout_rate_placeholder")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

net = tf.layers.conv2d(input_ph, 64, 5, 2, padding="VALID", activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 2048, int(net.get_shape()[1]), padding='VALID', activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training)
net = tf.layers.conv2d(net, 2048, 1, padding='VALID', activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training)
net = tf.layers.conv2d(net, num_variation_factor, 1,
                       bias_initializer=tf.zeros_initializer())
net = tf.squeeze(net, [1, 2])


print(net)
