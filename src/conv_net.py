import tensorflow as tf


class ConvNet():

    _session = None

    def __enter__(self):
        self._session = tf.Session()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._session.close()

    _y_hat = None
    _x = None

    #def Initalize_filters(self,X_train,kernel:int=3,amount_filters:int=1):
    #    n_C = X_train.shape[2]
    #    Convnet = tf.get_variable("Convnet",[kernel,kernel,n_C,amount_filters],
    #                              initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    #    return {"Convnet":Convnet}

    # def create_convnet(self,X_train,n_pixels:int=32,kernel:int=3,stride:int=1,filters:int=1):
    def fit(self,
            X, Y, width, height,
            filters: int = 1,
            kernel: int = 3,
            stride: int = 1,
            learning_rate: float = 0.05,
            training_epochs: int = 250):

        n_x = X.shape[0]
        n_y = Y.shape[0]

        # define variables
        self._x = x = tf.placeholder(tf.float32, [n_x, None], name="x")
        y = tf.placeholder(tf.float32, [n_y, None], name="y")

        x_res = tf.reshape(x, shape=[-1, width, height, 3])

        Conv1 = tf.layers.conv2d(x_res, filters, kernel, strides=(stride, stride))
        Conv1 = tf.layers.average_pooling2d(Conv1, 2, 2)

        FC = tf.contrib.layers.flatten(Conv1)
        FC = tf.layers.dense(FC, n_y)

        # define y_hat
        y_hat = tf.nn.sigmoid(FC)
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
