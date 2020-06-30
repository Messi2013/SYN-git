import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.name_scope('input'):
 
  input1 = tf.constant(3.0,name='A')
  input2 = tf.constant(4.0,name='B')
  input3 = tf.constant(5.0,name='C')
with tf.name_scope('op'):

  add = tf.add(input2,input3)

  mul = tf.multiply(input1,add)
with tf.Session() as ss:

  wirter = tf.summary.FileWriter('logs/',ss.graph)
  result = ss.run([mul,add])
  print(result)