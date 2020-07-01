#coding=utf-8
'''
Created on 2017-9-1
@author: wingdi
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##from sklearn.utils import shuffle
import pre_data
import tensorflow as tf

import model
from model import LeNet
import os

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

X_train,y_train,X_validation,y_validation,X_test,y_test = pre_data.pre_data()
##X_train, y_train = shuffle(X_train, y_train)
EPOCHS = 10
BATCH_SIZE = 128

##x = tf.compat.v1.placeholder(tf.float32, (None, 32, 32, 1))
x = tf.compat.v1.placeholder(tf.float32, (BATCH_SIZE, 32, 32, 1))
y = tf.compat.v1.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.compat.v1.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.compat.v1.get_default_session()
    sum_time_cost = 0.0
    sum_samples = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        if len(batch_x) < BATCH_SIZE:
            print("      skip last short batch in evaluate() ")
            break;  # skip the last short batch
        import time
        tm1 = time.time()
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        tm2 = time.time()
        total_accuracy += (accuracy * len(batch_x))
        sum_time_cost += tm2 - tm1
        sum_samples += BATCH_SIZE
    print("      samples %d cost %.6f or %.3f us per sample" % (
        sum_samples, sum_time_cost, sum_time_cost/sum_samples*1000*1000 )) # 75us on i7-6700hq
    return total_accuracy / num_examples

if False: # set to True to train and save
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        ##X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            if len(batch_x) < BATCH_SIZE:
                print("      skip last short batch in main ")
                break; # skip the last short batch
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, 'lenet')
    print("Model saved")
    
with tf.compat.v1.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

