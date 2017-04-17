
'''Palindromic Model'''

import numpy as np
import tensorflow as tf

binary_encode = lambda i,num_digits: np.array([i >> d & 1 for d in range(num_digits)])

def palindromic_encode(i):
    if str(i)[::-1] == str(i): return np.array([0,0,1])
    elif str(i)[::-1] == str(i): return np.array([0,1,0])
    else: return np.array([1,0,0])

NUM_DIGITS = 10
trX = np.array([binary_encode(i,NUM_DIGITS) for i in range(101,2**NUM_DIGITS)])
trY = np.array([palindromic_encode(i) for i in range(101,2**NUM_DIGITS)])

NUM_HIDDEN = 100
X = tf.placeholder('float',[None,NUM_DIGITS])
Y = tf.placeholder('float',[None,3])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

w_h = init_weights([NUM_DIGITS,NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN,3])

def model(X,w_h,w_o):
    h = tf.nn.relu(tf.matmul(X,w_h))
    return tf.matmul(h,w_o)

py_x = model(X,w_h,w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=py_x))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x,1)

def palindromic(i,prediction):
    m = 'This numbers is a palindromic (%s)'
    return [str(i), m % (i), m % (i)][prediction]

BATCH_SIZE = 512

if __name__ == '__main__':
  with tf.Session() as sess:
      tf.initialize_all_variables().run()

      for epoch in range(10**4):
          p = np.random.permutation(range(len(trX)))
          trX,trY = trX[p],trY[p]

          for start in range(0,len(trX),BATCH_SIZE):
              end = start + BATCH_SIZE
              sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end]})
              print(epoch,np.mean(np.argmax(trY,axis=1) == sess.run(predict_op,feed_dict={X:trX,Y:trY})))

          numbers = np.arange(1,1001)
          teX = np.transpose(binary_encode(numbers,NUM_DIGITS))
          teY = sess.run(predict_op,feed_dict={X:teX})
          output = np.vectorize(palindromic)(numbers,teY)
          print(output)
