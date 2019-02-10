import tensorflow as tf


class LogisticVectorRegression:

    _session = None

    def __enter__(self):
        self._session = tf.Session()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._session.close()

    _y_hat = None
    _x = None

    def fit(self,
            X, Y,
            learning_rate:float=0.05,
            training_epochs:int=250):

        x_n = X.shape[0]
        y_n = Y.shape[0]

        # define variables
        self._x = x = tf.placeholder(tf.float32, [x_n, None])
        y = tf.placeholder(tf.float32, [y_n, None])

        # define weights
        W = tf.Variable(tf.zeros([y_n, x_n]))
        b = tf.Variable(tf.zeros([y_n, 1]))

        # define y_hat
        z = tf.add(tf.matmul(W, x), b)
        y_hat = tf.nn.sigmoid(z)
        self._y_hat = y_hat

        # define cost
        cost = tf.norm(y - y_hat)

        # gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        self._session.run(init)

        # train
        for epoch in range(training_epochs):
            _, c = self._session.run([optimizer, cost], feed_dict={x: X, y: Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.4f}".format(c))

    def predict(self, X):
        return self._session.run(self._y_hat, feed_dict={self._x: X})


