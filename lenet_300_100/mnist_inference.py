import tensorflow as tf
from mnist.tf_mnist.mnist_train import variable_summaries
# LeNet-300-100
INPUT_NODE = 784                # 输入层节点数
OUTPUT_NODE = 10                # 输出层节点数

LAYER1_NODE = 300               # 隐藏层1节点数
LAYER2_NODE = 100               # 隐藏层2节点数


def get_weight_variable(layer_name, shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    variable_summaries(weights, layer_name+'/weights')
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, keep_prob, regularizer):
    with tf.variable_scope('layer1'):
        # 初始化权重和偏置
        with tf.name_scope('weights'):
            weights = get_weight_variable('layer1', [INPUT_NODE, LAYER1_NODE], regularizer)
        with tf.name_scope('biases'):
            biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
            variable_summaries(biases, 'layer1/biases')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input_tensor, weights) + biases
            variable_summaries(Wx_plus_b, 'layer1/Wx_plus_b')
        layer1 = tf.nn.relu(Wx_plus_b)
        layer1 = tf.nn.dropout(layer1, keep_prob)
        tf.summary.histogram('layer1_output', layer1)

    with tf.variable_scope('layer2'):
        # 初始化权重和偏置
        weights = get_weight_variable('layer2', [LAYER1_NODE, LAYER2_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER2_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
        layer2 = tf.nn.dropout(layer2, keep_prob)

    with tf.variable_scope('layer3'):
        # 初始化权重和偏置
        weights = get_weight_variable('layer3', [LAYER2_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer3 = tf.matmul(layer2, weights) + biases

    return layer3
