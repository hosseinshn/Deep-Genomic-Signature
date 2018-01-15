import numpy as np
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Load Your data

def PlaceH(n_input): 
    x = tf.placeholder(u"float", [None, n_input])
    y_ = tf.placeholder(u"float", [None,n_input])
    return x, y_

def corruption(input, corruption_level): #corruption of the input
    mask=np.random.binomial(1, 1 - corruption_level,input.shape ) #mask with several zeros at certain position
    corrupted_input=input*mask
    return corrupted_input

def Initializarion(n_input, n_hidden, n_samp):
    Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -4.0* math.sqrt(6/(n_input+n_samp)), 
                                        4.0* math.sqrt(6/(n_samp+n_input))))
    bh = tf.Variable(tf.zeros([n_hidden]))     
    Wo = tf.transpose(Wh) # tied weights
    bo = tf.Variable(tf.zeros([n_input]))
    return Wh, bh, Wo, bo

def AE(dat, Wh, Wo, bh, bo):     
    h = tf.nn.sigmoid(tf.matmul(dat,Wh) + bh)
    ho = tf.nn.sigmoid(tf.matmul(h, Wo) + bo)
    return h,ho

def KF(dat, k):
    kf = KFold(n_splits=k)
    kf.get_n_splits(dat)
    return kf.split(dat)

def compute_cost(Z, y_):
    logits = Z
    labels = y_    
    cost = tf.reduce_mean(tf.losses.mean_squared_error (predictions = logits , labels = labels))    
    return cost

def random_mini_batches(X, Y, mini_batch_size = 10, seed = 42):    
    np.random.seed(seed)            
    m = X.shape[0]                  
    mini_batches = []
        
    permutation = list(np.random.permutation(m))
    shuffled_X = X
    shuffled_Y = Y

    num_complete_minibatches = math.floor(m/mini_batch_size)
    
    #print(type(num_complete_minibatches))
    
    for k in xrange(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    Lower = num_complete_minibatches * mini_batch_size
    Upper = m - (mini_batch_size * math.floor(m/mini_batch_size))
    if m % mini_batch_size != 0:
        mini_batch_X = X[Lower : Lower + Upper, :]
        mini_batch_Y = Y[Lower : Lower + Upper, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches, permutation

def model(X_train, Y_train, X_test, Y_test, X_Test, Y_Test, learning_rate = 0.01,
          num_epochs = 10, minibatch_size = 10, print_cost = True):
    seed = 0 
    tf.set_random_seed(seed)    
    (n_samp, n_input) = X_train.shape                         
    costs = []                                        
    
    X, Y = PlaceH(n_input)

    Wh, bh, Wo, bo = Initializarion(n_input, n_hidden, n_samp)
    
    H, Z = AE(X, Wh, Wo, bh, bo)
    
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in xrange(num_epochs):

            epoch_cost = 0.                     
            num_minibatches = int(n_samp / minibatch_size) 
            seed = seed + 1
            minibatches, permutation = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost = epoch_cost + (minibatch_cost / num_minibatches)

            print u"Cost after epoch %i: %f" % (epoch, epoch_cost)
            costs.append(np.mean(epoch_cost))
                
        plt.plot(np.squeeze(costs))
        plt.ylabel(u'cost')
        plt.xlabel(u'iterations (per tens)')
        plt.title(u"Learning rate =" + unicode(learning_rate))
        plt.show()
        
        Weight = sess.run(Wh, feed_dict={X: X_train, Y: Y_train})
        bias = sess.run(bh, feed_dict={X: X_train, Y: Y_train})
        Hidden = sess.run(H, feed_dict={X: X_train, Y: Y_train})
        Y_hat = sess.run(Z, feed_dict={X: X_train, Y: Y_train})
        ts_cost = cost.eval({X: X_test, Y: Y_test})
        Ts_cost = cost.eval({X: X_Test, Y: Y_Test})
        print u"Dev cost:", ts_cost  
        print u"Test cost:", Ts_cost  
        print u"Parameters have been trained!"
        return Weight, bias, Hidden, Y_hat, costs, ts_cost, Ts_cost 
		
n_hidden = 256
corruption_level = 0.2
n_fold = 10
final_train_cost = []
final_test_cost = []
final_params = []
final_permutations = []

output_data = data
input_data = corruption(data, corruption_level)


n_samp, n_input = input_data.shape 

u"""
folds = KF(input_data, n_fold)
for tr, ts in folds: 
     parameters, tr_costs, ts_costs, permutation = DAE.model(input_data[tr, :], output_data[tr, :], input_data [ts, :], 
                                                            output_data[ts, :],  learning_rate = 0.01, 
                                                            num_epochs = 35, minibatch_size = 10,)
     final_params.append(parameters)
     final_train_cost.append(tr_costs)
     final_test_cost.append(ts_costs)
     final_permutations.append(permutation)
"""

from __future__ import absolute_import
Weight, bias, Hidden, Y_hat, tr_costs, ts_costs, Ts_cost = model(input_data, output_data, Dev, 
                                                            Dev, Test, Test, learning_rate = 0.05, 
                                                            num_epochs = 1000, minibatch_size = 10)
print "Files are saved!"