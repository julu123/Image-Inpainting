import tensorflow as tf
import math

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
            batch_size,
            learning_rate: float = 0.0001,
            training_epochs: int = 250):

        n_x = X.shape[0]
        n_y = Y.shape[0]
        m = X.shape[1]

        kernel = 8
        stride = kernel

        # define variables
        self._x = x = tf.placeholder(tf.float32, [None, n_x], name="x")
        y = tf.placeholder(tf.float32, [None, n_y], name="y")

        x_res = tf.reshape(tf.transpose(x), shape=[-1, width, height, 3])

        # conv = tf.layers.conv2d(x_res,
        #                        filters=32,
        #                        kernel_size=kernel,
        #                        use_bias=False,
        #                        strides=(stride, stride))

        FC = tf.contrib.layers.flatten(x_res)
        FC = tf.layers.dense(FC, n_y, use_bias=False)

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
            for batch in range(int(m/batch_size)):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X[:, start:end]
                Y_batch = Y[:, start:end]
                _, c = self._session.run([optimizer, cost], feed_dict={x: X_batch.T, y: Y_batch.T})
                print("Epoch:", '%04d' % (epoch + 1), "batch:", '%04d' % (batch + 1), "cost: ", "{:.4f}".format(c))

    def predict(self, X):
        return self._session.run(self._y_hat, feed_dict={self._x: X.T}).T
