import tensorflow as tf
from Model_OCL import *
from run_OCL import split_mnist,rotate_mnist,shuffle_mnist
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.enable_eager_execution()

def train(task_list,task_label):
    learning_rate = 0.1
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    Model = OCL_Net('mlp',num_classes=10,dim = dim, seed_num=11, optimizer=optimizer)
    




num_task = 5

num_classes = 10
dim = [28*28]
sequence_length=np.prod(dim)
mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

np.random.seed(11)

runs = []
runs_train=[]
runs_loss=[]
runs_label=[]

task_labels = np.arange(num_classes)

np.random.shuffle(task_labels)
runs_label.append(task_labels)

#pmnist
task_list=[]
task_labels = [np.arange(num_classes)]*num_task
for i in range(num_task):
    task_list.append(shuffle_mnist(mnist,i+num_task))

acculist,losslist,train_acculist = train(task_list,task_labels)


    