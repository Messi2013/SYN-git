import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input1=tf.constant([1.0,2.0,3.0],name="input1")
input2=tf.Variable(tf.random_uniform([3]),name="input")
output=tf.add_n([input1,input2],name="add")


writer=tf.summary.FileWriter(".\log",tf.get_default_graph())
writer.close()
init = tf.global_variables_initializer()

tf.Session().run(init)
