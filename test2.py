import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
x=tf.compat.v1.placeholder(tf.float32,[None,1])
W=tf.compat.v1.Variable(tf.zeros([1,1]))
b=tf.compat.v1.Variable(tf.zeros([1]))
y=tf.compat.v1.matmul(x,W)+b
y_=tf.compat.v1.placeholder(tf.float32,[None,1])
cost=tf.compat.v1.reduce_sum(tf.pow((y_-y),2))

train_step=tf.compat.v1.train.GradientDescentOptimizer(0.0001).minimize(cost)
tf.summary.histogram('cost', cost)
init = tf.compat.v1.initialize_all_variables()
sess = tf.compat.v1.Session()
sess.run(init)
merged_summary_op = tf.compat.v1.summary.merge_all()  # 合并所有的summary
summary_writer = tf.compat.v1.summary.FileWriter('log/linear', sess.graph)
for i in range(100):
    xs=np.array([[i]])
    ys=np.array([[2*i]])
    feed={x:xs,y_:ys}
    sess.run(train_step,feed_dict=feed)
    print("Iterator:%d"%i)
    print("W=%f"%sess.run(W))
    print("b=%f"%sess.run(b))
