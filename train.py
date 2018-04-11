import tensorflow as tf

batch_size = 2
input_size = 160
num_channel = 3
learning_rate = 0.01
num_variation_factor = 3
opt_epsilon = 1.
rmsprop_momentum = 0.9
rmsprop_decay = 0.9

inputs_ph = tf.placeholder(tf.float32, [batch_size, input_size, input_size, num_channel], "inputs_placeholder")
labels_ph = tf.placeholder(tf.float32, [batch_size, num_variation_factor], "labels_placeholder")
dropout_rate_ph = tf.placeholder(tf.float32, (), "dropout_rate_placeholder")
weight_decay_ph = tf.placeholder(tf.float32, ())
is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

net = tf.layers.conv2d(inputs_ph, 64, 5, 2, padding="VALID", activation=tf.nn.relu,
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
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training_ph)
net = tf.layers.conv2d(net, 2048, 1, padding='VALID', activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training_ph)
net = tf.layers.conv2d(net, num_variation_factor, 1,
                       bias_initializer=tf.zeros_initializer())
logits = tf.squeeze(net, [1, 2])

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_ph, logits=logits),
                         name="sigmoid_cross_entropy")
print(loss_op)
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
regularizer = tf.constant(0., tf.float32)
for weight in weights:
    regularizer += tf.nn.l2_loss(weight)
regularizer *= weight_decay_ph
loss_op += regularizer
opt = tf.train.RMSPropOptimizer(
    learning_rate,
    decay=rmsprop_decay,
    momentum=rmsprop_momentum,
    epsilon=opt_epsilon)
train_op = opt.minimize(loss_op)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels_ph, 1)), tf.float32))
pred_idx_op = tf.argmax(logits, 1)
print(accuracy_op, pred_idx_op)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
