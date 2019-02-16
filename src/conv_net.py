import tensorflow as tf


class ConvNet:

    _session = None

    def __enter__(self):
        self._session = tf.Session()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._session.close()

    _y_hat = None
    _x = None

    # def create_convnet(self,X_train,n_pixels:int=32,kernel:int=3,stride:int=1,filters:int=1):
    def fit(self,
            X, Y, width, height,
            kernel: int = 5,
            stride: int = 1,
            learning_rate: float = 0.05,
            training_epochs: int = 250):

        n_x = X.shape[0]
        n_y = Y.shape[0]

        # define variables
        self._x = x = tf.placeholder(tf.float32, [None, n_x], name="x")
        y = tf.placeholder(tf.float32, [None, n_y], name="y")

        # x_res = tf.reshape(tf.transpose(x), shape=[-1, width, height, 3])

        # Conv1 = tf.layers.conv2d(x_res, 3, kernel, strides=(stride, stride))
        # Conv1 = tf.layers.average_pooling2d(Conv1, 2, 2)

        # Conv1 = tf.layers.average_pooling2d(x_res, 4, 4)

        # FC = tf.contrib.layers.flatten(Conv1)
        FC = tf.contrib.layers.flatten(x)
        FC = tf.layers.dense(FC, n_y)

        # define y_hat
        y_hat = tf.nn.sigmoid(FC)
        self._y_hat = y_hat

        # define cost
        cost = tf.norm(y - y_hat)

        # gradient descent
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        self._session.run(init)

        # train
        for epoch in range(training_epochs):
            _, c = self._session.run([optimizer, cost], feed_dict={x: X.T, y: Y.T})
            print("Epoch:", '%04d' % (epoch + 1), "cost: ", "{:.4f}".format(c))

    def predict(self, X):
        return self._session.run(self._y_hat, feed_dict={self._x: X.T}).T
