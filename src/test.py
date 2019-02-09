import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf

#path = '/Users/justinlundgren/desktop/Image0.jpg'
#im = Image.open(path)
#im.show()

#def Initlialize(x,y):
#    n_x = len(x)
#    n_y = len(y)
#   W1 = np.random.rand(n_y,n_x)*0.01
#    b1 = np.zeros((n_y,1))
#    return W1,b1

#def Activate(W1,b1,x):
#   yhat = tf.sigmoid(tf.add(tf.matmul(W1,x),b1))
#   sess = tf.Session()
#   result = sess.run(yhat)
#   sess.close()
#   return result

#def cost(y,yhat):
#   real = tf.placeholder(tf.float32, name="real")
#   pred = tf.placeholder(tf.float32, name="pred")
#   cost = tf.norm(y-yhat)
#   sess = tf.Session()
#   cost = sess.run(cost, feed_dict={y: real, yhat: pred})
#   sess.close()
#   return cost


def Multivariate_log_regression(X,Y,learning_rate:float=0.05,training_epochs:int=2500):#,batch_size:int=100,display_step:int=1):
    x_n=len(X)
    y_n=len(Y)

    #Define varibles
    x = tf.placeholder(tf.float32, [None, x_n])
    y = tf.placeholder(tf.float32, [None, y_n])

    #Define weights
    W = tf.Variable(tf.zeros([x_n, y_n]))
    b = tf.Variable(tf.zeros([y_n]))

    #Define yhat
    yhat = tf.nn.sigmoid(tf.add(tf.matmul(W,x),b))

    #Define cost
    cost = tf.norm(y-yhat)

    #Gradient Decent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        #Train
        for epoch in range(training_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={x: x, y: y})

