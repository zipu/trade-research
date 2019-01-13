import tensorflow as tf
import numpy as np

class CNN:

    def __init__(self, sess, name='trend', learning_rate=0.001, period=120, depth=3,
                 filters=32, kernel_size=5, pool_size=2, pooling=False):
        self.sess = sess
        self.name = name
        self._build_net(period, learning_rate, depth, filters, kernel_size, pool_size, pooling)
        

    def _build_net(self, period, learning_rate, depth, filters, ksize, pool_size, pooling):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
            self.training = tf.placeholder(tf.bool)
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, period, 4])
            # periodx4 (OHLC), Input Layer
            ohlc = tf.reshape(self.X, [-1, period,  4])
            # 3(down trend, neither, up trend), ouput layer
            self.Y = tf.placeholder(tf.float32, [None, 3])

            convs = []
            pools = []
            dropouts = []
            for step in range(depth):
                inputs = convs[step-1] if convs else ohlc
                # Convolutional Layer
                conv = tf.layers.conv1d(inputs=inputs, filters=filters*pow(2,step), kernel_size=ksize,\
                                              padding="same", activation=tf.nn.relu)
                convs.append(conv)
                if pooling:
                    # Pooling Layer
                    pool = tf.layers.max_pooling1d(inputs=conv[step], pool_size=pool_size, padding="same", strides=1)
                    pools.append(pool)
                    input_layer = pool
                else:
                    input_layer = conv
                
                dropout = tf.layers.dropout(inputs=input_layer, rate=0.7, training=self.training)
                dropouts.append(dropout)
            # Dense Layer with Relu
            size = np.prod(dropouts[-1].shape.as_list()[1:])
            flat = tf.reshape(dropouts[-1], [-1, size])
            
            dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
            dropout_dense = tf.layers.dropout(inputs=dense, rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 1024 inputs -> 3 outputs
            self.logits = tf.layers.dense(inputs=dropout_dense, units=3)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})