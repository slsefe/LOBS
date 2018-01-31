import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from LOBS.lenet_300_100 import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8        # 基础学习率
LEARNING_RATE_DECAY = 0.9       # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 正则化损失系数
TRAINING_STEPS = 20000          # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率
# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'Saved_model/'
MODEL_NAME = 'model.ckpt'
SUMMARY_DIR = 'tmp/log2'

# 生成变量监控信息，生成监控信息日志，var表示需要记录的张量，name表示对应的图表名称
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        # 记录张量中元素的取值分布
        tf.summary.histogram(name, var)
        # 计算变量的平均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        # 计算变量的标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/'+name, stddev)

def train(mnist):
    with tf.name_scope('input'):
        '''定义输入'''
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    with tf.name_scope('input_reshape'):
        '''将输入向量还原为图片'''
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('regularizer'):
        # L2正则化
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        #tf.summary.scalar('regularizer', tf.cast(regularizer, dtype=tf.float32))

    y = mnist_inference.inference(x, 0.5, regularizer)
    # 存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('moving_average'):
        # 初始化滑动平均类
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        # tf.summary.scalar('moving_average', variable_averages)
        # 在所有神经网络参数的变量上使用滑动平均
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('cross_entropy'):
        # 计算整个minibatch的交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
        # 当前batch中所有样例的交叉熵平均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('losses'):
                # 总损失
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        # loss = cross_entropy_mean
        tf.summary.scalar('losses', loss)

    with tf.name_scope('train_step'):
        # 指数衰减的学习率
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
        # 每一次训练过程
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # 通过反向传播来更新参数、同时更新每个参数的滑动平均值 train_op = tf.group(train_step, variable_averages_op)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    # 初始化tf持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, tf.get_default_graph())
        # 初始化参数
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _, loss_value, step = sess.run([merged, train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            summary_writer.add_summary(summary, i)
            # 每1000轮保存一次模型
            if i % 1000 == 0 or i+1 ==TRAINING_STEPS:
                # 输出当前训练情况
                print('After %d training step(s), loss on training batch is %g' % (step, loss_value))
                # 保存当前模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

    summary_writer.close()


def main(arev=None):
    # 获取数据集，分别为：train.images 55000*784 train_labels 55000*10
    # test.images 10000*784 test.image 10000*10
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
