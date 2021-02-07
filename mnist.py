import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data

# W定义一个获取卷积核的函数（随机卷积核初始化）


def weight_variable(shape):
    # Outputs random values from a truncated normal distribution.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# b定义一个获取偏置值的函数


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义一个卷积函数


def conv2d(Input, fliter):
    return tf.nn.conv2d(Input, fliter, [1, 1, 1, 1], padding="SAME")

# 最大池化函数


def max_pool_2x2(value):
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


# Input->conv_1->relu_1->pool_1 ->conv_2->relu_2->pool_2->FullConnect(FC)
if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(shape=[None, 28*28], dtype=tf.float32)
    lable = tf.placeholder(shape=[None, 10], dtype=tf.float32)

    Input = tf.reshape(x, [-1, 28, 28, 1])
    
    # 第一个卷积层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(Input, W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 14*14*32
    
    # 第二个卷积层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # 7*7*64

    # 全连接层，输出为1024(512)维向量
    W_fc1 = weight_variable([7*7*64, 512])
    b_fc1 = weight_variable([512])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 把1024维向量转换成10维，对应10个类别
    W_fc2 = weight_variable([512, 10])
    b_fc2 = weight_variable([10])
    y_conv = tf.matmul(h_fc1, W_fc2)+b_fc2

    # 直接使用tf.nn.softmax_cross_entropy_with_logits直接计算交叉熵
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=lable, logits=y_conv))
    # 定义train_step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(lable, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # 文件写路径
    '''saver = tf.train.Saver()
    merged = tf.summary.merge_all()  # 合并
    summary_writer = tf.summary.FileWriter(
        './mnistEven/', graph=sess.graph)'''  

    # 训练
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i % 50 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], lable: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
#            summary = sess.run(merged, feed_dict={x: batch[0], lable: batch[1], keep_prob: 1.0}) 
#            summary_writer.add_summary(summary, i)  # 将所有搜集的写文件
        sess.run(train_step, feed_dict={
                     x: batch[0], lable: batch[1], keep_prob: 0.5})
#    print("test accuracy %g" % sess.run(accuracy, feed_dict={
#        x: mnist.test.images, lable: mnist.test.labels, keep_prob: 1.0}))#这个GPU跑不了


'''accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
good = 0
total = 0
for i in range(10):
    testSet = mnist.test.next_batch(50)
    good += accuracy_sum.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0})
    total += testSet[0].shape[0]
print ("test accuracy %g"%(good/total))'''
