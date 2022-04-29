

import math
import tensorflow as tf
import numpy as np
def _shuffle(x,seed):
    if seed==0:
        return x
    shape=x.get_shape().as_list()
    n = shape[-1]
    with tf.Session() as sess:
        shuffle_mat = sess.run(tf.random_shuffle(tf.eye(n),seed=seed))
    x = tf.reshape(x,[-1,n])
    x = tf.matmul(x,shuffle_mat)
    x = tf.reshape(x,[-1]+shape[1:])
    return x

def _conv(x, kernel_size, out_channels, stride, var_list, pad="SAME", name="conv"):
    """
    Define API for conv operation. This includes kernel declaration and
    conv operation both.
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        #n = kernel_size * kernel_size * out_channels
        n = kernel_size * in_channels
        stdv = 1.0 / math.sqrt(n)
        w = tf.get_variable('kernel', [kernel_size, kernel_size, in_channels, out_channels],
                           tf.float32, 
                           initializer=tf.random_uniform_initializer(-stdv, stdv))
                           #initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))

        # Append the variable to the trainable variables list
        var_list.append(w)

    # Do the convolution operation
    output = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=pad)
    return output

def _fc(x, out_dim, var_list, name="fc"):
    """
    Define API for the fully connected layer. This includes both the variable
    declaration and matmul operation.
    """
    in_dim = x.get_shape().as_list()[1]
    stdv = 1.0 / math.sqrt(in_dim)
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        # Define the weights and biases for this layer
        w = tf.get_variable('weights', [in_dim, out_dim], tf.float32, 
                initializer=tf.random_uniform_initializer(-stdv, stdv))
                #initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [out_dim], tf.float32, initializer=tf.constant_initializer(0))

        # Append the variable to the trainable variables list
        var_list.append(w)
        var_list.append(b)

    # Do the FC operation
    output = tf.matmul(x, w) + b
    return output

def _bn(x, var_list, train_phase, name='bn_'):
    """
    Batch normalization on convolutional maps.
    Args:

    Return:
    """
    #return x 
    n_out = x.get_shape().as_list()[3]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        beta = tf.get_variable('beta', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        var_list.append(beta)
        var_list.append(gamma)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed

def _residual_block(x, trainable_vars, train_phase, apply_relu=True, name="unit",taskID=0):
    """
    ResNet block when the number of channels across the skip connections are the same
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        shortcut = x
        x = _conv(x, 3, in_channels, 1, trainable_vars, name='conv_1')
        x = _bn(x, trainable_vars, train_phase, name="bn_1")
        x = tf.nn.relu(x)
        x = _shuffle(x,taskID)
        x = _conv(x, 3, in_channels, 1, trainable_vars, name='conv_2')
        x = _bn(x, trainable_vars, train_phase, name="bn_2")
        x = _shuffle(x,taskID)
        x = x + shortcut
        if apply_relu == True:
            x = tf.nn.relu(x)

    return x

def _residual_block_first(x, out_channels, strides, trainable_vars, train_phase, apply_relu=True, name="unit", taskID=0):
    """
    A generic ResNet Block
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        # Figure out the shortcut connection first
        if in_channels == out_channels:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = _conv(x, 1, out_channels, strides, trainable_vars, name="shortcut")
            shortcut = _bn(shortcut, trainable_vars, train_phase, name="bn_0")

        # Residual block
        x = _conv(x, 3, out_channels, strides, trainable_vars, name="conv_1")
        x = _bn(x, trainable_vars, train_phase, name="bn_1")
        x = tf.nn.relu(x)
        x = _shuffle(x,taskID)
        x = _conv(x, 3, out_channels, 1, trainable_vars, name="conv_2")
        x = _bn(x, trainable_vars, train_phase, name="bn_2")
        x = _shuffle(x,taskID)

        x = x + shortcut
        if apply_relu:
            x = tf.nn.relu(x)

    return x


# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
from RGO_Optimizer import RGO_Optimizer
import numpy as np

NEG_INF = -1e32

class SGD_Net(object):
    def __init__(self, arch='resnet18',num_classes = 100,dim = [32,32,3],optimizer=tf.train.MomentumOptimizer(0.01, momentum=0.9), seed_num=0):
        seed(seed_num)
        set_random_seed(seed_num)
        SEED=seed_num
        # Placeholders for input, output and dropout
        self.dim = dim
        sequence_length = np.prod(dim)
        num_classes = num_classes
        self.arch = arch
        self.total_classes = num_classes
        #used for batch_norm or other same architecturesd
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.input_x = tf.placeholder(tf.float32, [None]+dim, name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
        self.output_mask = tf.placeholder(dtype=tf.float32, shape=[num_classes])
        #self.output_mask = tf.ones(shape=[num_classes])
        
        self._optimizer=optimizer
        self.optimizer_initialized=False
        self.creat_ops(0)

    def shuffle(self,x,seed):
        return x
    def get_ops(self,taskID=0):
        return [self.accuracy,self.loss,self.back_forward]
    def creat_ops(self,taskID):
        taskID = 0
        #always use no encoding layer.
        if self.arch =='resnet18':
            images=tf.reshape(self.input_x,[-1]+self.dim)
            kernels = [7, 3, 3, 3, 3]
            filters = [64, 64, 128, 256, 512]
            strides = [1, 0, 2, 2, 2]
            scores = self.resnet18_conv_feedforward(images, kernels, filters, strides,taskID)
        elif self.arch == 'mlp':
            hiddens = [256,256]
            scores = self.mlp_feedforward(self.input_x,hiddens,taskID)
        elif self.arch == 'lenet':
            images=tf.reshape(self.input_x,[-1]+self.dim)
            scores = self.lenet_feedforward(images,taskID)
        elif 'vgg' in self.arch:
            images=tf.reshape(self.input_x,[-1]+self.dim)
            scores = self.vgg_feedforward(images,self.arch,taskID)
        elif self.arch == 'alexnet':
            images=tf.reshape(self.input_x,[-1]+self.dim)
            scores = self.alexnet_feedforward(images,taskID)
        elif self.arch == 'alexnet2':
            images=tf.reshape(self.input_x,[-1]+self.dim)
            scores = self.alexnet2_feedforward(images,taskID)

        pruned_logits = tf.where(tf.tile(tf.equal(self.output_mask[None,:], 1.0), [tf.shape(scores)[0], 1]), scores, NEG_INF*tf.ones_like(scores))

        if not self.optimizer_initialized:
            self.optimizer =  self._optimizer
            self.optimizer_initialized=True
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, 
                logits=pruned_logits)
        #losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
        loss = tf.reduce_mean(losses)

        back_forward = self.optimizer.minimize(loss,var_list=self.trainable_vars)
        self.update=[]

        predictions = tf.argmax(pruned_logits, 1, name="predictions")
        correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        self.accuracy,self.loss,self.back_forward = accuracy,loss,back_forward
        return [accuracy,loss,back_forward]

    def resnet18_conv_feedforward(self, h, kernels, filters, strides, taskID):
        """
        Forward pass through a ResNet-18 network

        Returns:
            Logits of a resnet-18 conv network
        """
        self.trainable_vars = []

        # Conv1
        h = _conv(h, kernels[0], filters[0], strides[0], self.trainable_vars, name='conv_1')
        h = _bn(h, self.trainable_vars, self.train_phase, name='bn_1')
        h = tf.nn.relu(h)
        h=self.shuffle(h,taskID)
        # Conv2_x
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv2_1',taskID=taskID)
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv2_2',taskID=taskID)
        h=self.shuffle(h,taskID)
        # Conv3_x
        h = _residual_block_first(h, filters[2], strides[2], self.trainable_vars, self.train_phase, name='conv3_1',taskID=taskID)
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv3_2',taskID=taskID)
        h=self.shuffle(h,taskID)
        # Conv4_x
        h = _residual_block_first(h, filters[3], strides[3], self.trainable_vars, self.train_phase, name='conv4_1',taskID=taskID)
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv4_2',taskID=taskID)
        h=self.shuffle(h,taskID)
        # Conv5_x
        h = _residual_block_first(h, filters[4], strides[4], self.trainable_vars, self.train_phase, name='conv5_1',taskID=taskID)
        h = _residual_block(h, self.trainable_vars, self.train_phase, name='conv5_2',taskID=taskID)
        h=self.shuffle(h,taskID)
        # Apply average pooling
        h = tf.reduce_mean(h, [1, 2])
        # Store the feature mappings
        self.features = h
        self.image_feature_dim = h.get_shape().as_list()[-1]
        h=self.shuffle(h,taskID)
        logits = _fc(h, self.total_classes, self.trainable_vars, name='fc_1')
        return logits

    def mlp_feedforward(self,images,hiddens,taskID):
        self.trainable_vars = []
        out = images
        for i,hidden in enumerate(hiddens):
            out = _fc(out, hidden,self.trainable_vars,name='fc_'+str(i+1))
            out = tf.nn.relu(out)
            out = self.shuffle(out,taskID)
        out = _fc(out, self.total_classes,self.trainable_vars,name='fc_'+str(len(hiddens)+1))
        return out
    
    def lenet_feedforward(self, h, taskID):
        self.trainable_vars = []

        h = _conv(h, 5, 20, 1 , self.trainable_vars, name='conv_1')
        h = tf.nn.relu(h)
        h = self.shuffle(h,taskID)
        h = _conv(h, 5, 50, 1 , self.trainable_vars, name='conv_2')
        h = tf.nn.relu(h)

        shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, shape[1] * shape[2] * shape[3]])

        h = self.shuffle(h,taskID)
        h = _fc(h,800,self.trainable_vars, name='fc_1')
        h = tf.nn.relu(h)
        h = self.shuffle(h,taskID)
        h = _fc(h,500,self.trainable_vars, name='fc_2')
        h = tf.nn.relu(h)
        h = self.shuffle(h,taskID)
        h = _fc(h,self.total_classes, self.trainable_vars, name='fc_3')
        return h
    def alexnet_feedforward(self, h, taskID):
        self.trainable_vars = []

        h = _conv(h, 4, 64, 1 , self.trainable_vars, name='conv_1')
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        h = self.shuffle(h,taskID)

        h = _conv(h, 3, 128, 1 , self.trainable_vars, name='conv_2')
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        h = self.shuffle(h,taskID)

        h = _conv(h, 2, 256, 1 , self.trainable_vars, name='conv_3')
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, shape[1] * shape[2] * shape[3]])

        h = self.shuffle(h,taskID)
        h = _fc(h,2048,self.trainable_vars, name='fc_1')
        h = tf.nn.relu(h)
        h = self.shuffle(h,taskID)
        h = _fc(h,2048,self.trainable_vars, name='fc_2')
        h = tf.nn.relu(h)
        h = self.shuffle(h,taskID)
        h = _fc(h,self.total_classes, self.trainable_vars, name='fc_3')
        return h
    def vgg_feedforward(self, h,arch, taskID):
        self.trainable_vars = []
        vgg_cfg = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        cfg = vgg_cfg[arch]
        for i,x in enumerate(cfg):
            if x == 'M':
                h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
            else:
                h = _conv(h, 3, x, 1 , self.trainable_vars, name='conv_'+str(i))
                h = tf.nn.relu(h)
                h = self.shuffle(h,taskID)

        shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, shape[1] * shape[2] * shape[3]])
        h = self.shuffle(h,taskID)
        h = _fc(h,self.total_classes, self.trainable_vars, name='fc_3')
        return h
    def alexnet2_feedforward(self, h, taskID):
        self.trainable_vars = []

        h = _conv(h, 5, 64, 1 , self.trainable_vars, name='conv_1')
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        h = self.shuffle(h,taskID)

        h = _conv(h, 4, 128, 1 , self.trainable_vars, name='conv_2')
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        h = self.shuffle(h,taskID)

        h = _conv(h, 3, 128, 1 , self.trainable_vars, name='conv_3')
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        h = self.shuffle(h,taskID)

        h = _conv(h, 3, 128, 1 , self.trainable_vars, name='conv_4')
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, shape[1] * shape[2] * shape[3]])

        h = self.shuffle(h,taskID)
        h = _fc(h,1024,self.trainable_vars, name='fc_1')
        h = tf.nn.relu(h)
        h = self.shuffle(h,taskID)
        h = _fc(h,256,self.trainable_vars, name='fc_2')
        h = tf.nn.relu(h)
        h = self.shuffle(h,taskID)
        h = _fc(h,self.total_classes, self.trainable_vars, name='fc_3')
        return h
    def future_arch_feedforward(self,h,taskID):
        # if you want to add new archs, 
        # just add a line of "h = self.shuffle(h,taskID)" after each sublayer of origin single task model.
        pass





