# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
# import tf.keras.datasets.mnist as DataSet
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import sys
from copy import deepcopy
# from test_cifar import get_data_set
import pickle
import scipy
import argparse
import random
from data_utils import construct_split_miniImagenet
from six.moves.urllib.request import urlretrieve
import tarfile
import zipfile

tf.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu

VALID_OPTIMS = ['SGD', 'MOMENTUM', 'ADAM']
VALID_ARCHS = ['mlp', 'lenet', 'resnet18']
VALID_MODELS = ['SGD','MTL','STL','RLL','OCL']
VALID_DATASETS = ['p-MNIST','r-MNIST','CIFAR','miniImageNet']

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for RLL experiment.")
    # basic setting
    parser.add_argument("--method", type=str, default='RLL',
                       help="Model to be used for LLL. \
                        \n \nSupported values: %s"%(VALID_MODELS))
    parser.add_argument("--arch", type=str, default='resnet18',
                        help="Network Architecture for the experiment.\
                                \n \nSupported values: %s"%(VALID_ARCHS))
    parser.add_argument("--dataset", type=str, default='CIFAR',
                       help="\n \nSupported values: %s"%(VALID_DATASETS))
    parser.add_argument("--seed", type=int, default=79,
                       help="Random Seed.")
    parser.add_argument("--gpuid", type=int, default=0,
                       help="GPU used.")

    # training details
    parser.add_argument("--num-runs", type=int, default=5,
                       help="Total runs/ experiments over which accuracy is averaged.")
    parser.add_argument("--epoch", type=int, default=8,
                       help="Number of epochs for each task.")
    parser.add_argument("--maxstep", type=int, default=-1,
                       help="max steps for each task.")
    parser.add_argument("--num-task", type=int, default=20,
                       help="Number of tasks.")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Mini-batch size for each task.")
    parser.add_argument("--optim", type=str, default='SGD',
                        help="Optimizer for the experiment. \
                                \n \nSupported values: %s"%(VALID_OPTIMS))
    parser.add_argument("--learning-rate", type=float, default=0.03,
                       help="Starting Learning rate for each task.")
    
    # log settings
    parser.add_argument("--log-dir", type=str, default='./log/',
                       help="Directory where the plots and model accuracies will be stored.")
    return parser.parse_args()

# ==================================================












# ==================================================
CIFAR_10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100_URL = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR_10_DIR = "/cifar_10"
CIFAR_100_DIR = "/cifar_100"

def split_mnist(mnist, cond):
    sets = ["train", "validation", "test"]
    sets_list = []
    for set_name in sets:
        this_set = getattr(mnist, set_name)
        maxlabels = np.argmax(this_set.labels, 1)
        sets_list.append(DataSet(this_set.images[cond(maxlabels),:], this_set.labels[cond(maxlabels)],
                                 dtype=dtypes.uint8, reshape=False))
    return base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])

def rotate_mnist(mnist,angle):
    print('rotating mnist with angle:'+str(angle))
    sets = ["train", "validation", "test"]
    sets_list = []
    for set_name in sets:
        this_set = getattr(mnist, set_name)
        maxlabels = np.argmax(this_set.labels, 1)
        a=this_set.images[:,:].copy()
        # print(a.shape)
        for i in range(np.shape(a)[0]):
            a[i]=scipy.ndimage.interpolation.rotate(a[i,:].reshape(28,28),angle, order=1, reshape=False).reshape(784)
        sets_list.append(DataSet(a, this_set.labels,
                                 dtype=dtypes.uint8, reshape=False))
    return base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])


def shuffle_mnist(mnist,seednum):
    print('permuting mnist with seed:'+str(seednum))
    ss = np.arange(28 * 28)
    np.random.seed(seednum)
    if seednum > 0:
        #np.random.seed(seednum)
        np.random.shuffle(ss)

    sets = ["train", "validation", "test"]
    sets_list = []

    copied_mnist=deepcopy(mnist)

    for set_name in sets:
        this_set = getattr(copied_mnist, set_name)
        perm = np.arange(len(this_set.labels))
        np.random.shuffle(perm)
        a=this_set.images[:,ss]
        sets_list.append(DataSet(a[perm,:], this_set.labels[perm],
                                 dtype=dtypes.uint8, reshape=False))
    return base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])


def _cifar_maybe_download_and_extract(data_dir):
    """
    Routine to download and extract the cifar dataset

    Args:
        data_dir      Directory where the downloaded data will be stored
    """
    cifar_10_directory = data_dir + CIFAR_10_DIR
    cifar_100_directory = data_dir + CIFAR_100_DIR

    # If the data_dir does not exist, create the directory and download
    # the data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

        url = CIFAR_10_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_10 = file_path
        print("Downloading CIFAR10.")
        file_path, _ = urlretrieve(url=url, filename=file_path)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")

        url = CIFAR_100_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_100 = file_path
        print("Downloading CIFAR100.")
        file_path, _ = urlretrieve(url=url, filename=file_path)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")

        os.rename(data_dir + "/cifar-10-batches-py", cifar_10_directory)
        os.rename(data_dir + "/cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_10)
        os.remove(zip_cifar_100)

def construct_split_cifar100(task_labels):

    data_dir = 'data/CIFAR_data'

    # Get the cifar dataset
    cifar_data = _get_cifar(data_dir)

    # Define a list for storing the data for different tasks
    datasets = []

    # Data splits
    sets = ["train", "test"]

    def dense_to_big_hot(labels_dense):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * 100
        labels_one_hot = np.zeros((num_labels, 100))
        labels_one_hot.flat[index_offset + labels_dense.ravel()%100] = 1

        return labels_one_hot
    validation=[]

    for task in task_labels:

        for set_name in sets:
            this_set = cifar_data[set_name]
            global_class_indices = np.column_stack( np.nonzero(dense_to_big_hot(this_set[1])))
            #print(global_class_indices)
            #print(this_set[1])
            count = 0
            for cls in task:
                if count == 0:
                    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])])
                else:
                    class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                                 cls][:,np.array([True, False])]))
                count += 1

            class_indices = np.sort(class_indices, axis=None)

            def dense_to_one_hot(labels_dense):
                num_labels = labels_dense.shape[0]
                index_offset = np.arange(num_labels) * 10
                labels_one_hot = np.zeros((num_labels, 10))
                labels_one_hot.flat[index_offset + labels_dense.ravel()%10] = 1

                return labels_one_hot

            if set_name == "train":
                train = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    # 'labels':dense_to_one_hot( deepcopy(this_set[1][class_indices]) ),
                    # 'biglabels':dense_to_big_hot(deepcopy(this_set[1][class_indices])),
                    'labels':dense_to_big_hot(deepcopy(this_set[1][class_indices])),
                }
            elif set_name == "validation":
                validation = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    # 'labels':dense_to_one_hot(deepcopy(this_set[1][class_indices])),
                    # 'biglabels':dense_to_big_hot(deepcopy(this_set[1][class_indices])),
                    'labels':dense_to_big_hot(deepcopy(this_set[1][class_indices])),
                }
            elif set_name == "test":
                test = {
                    'images':deepcopy(this_set[0][class_indices, :]),
                    # 'labels':dense_to_one_hot(deepcopy(this_set[1][class_indices])),
                    # 'biglabels':dense_to_big_hot(deepcopy(this_set[1][class_indices])),
                    'labels':dense_to_big_hot(deepcopy(this_set[1][class_indices])),
                }

        cifar = {
            'train': train,
            'validation': validation, 
            'test': test,
        }

        datasets.append(cifar)

    return datasets


def _get_cifar(data_dir):
    """
    Get the CIFAR-10 and CIFAR-100 datasets

    Args:
        data_dir        Directory where the downloaded data will be stored
    """
    x_train = None
    y_train = None
    x_validation = None
    y_validation = None
    x_test = None
    y_test = None

    def dense_to_one_hot(labels_dense, num_classes=100):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()%num_classes] = 1

        return labels_one_hot

    # Download the dataset if needed
    _cifar_maybe_download_and_extract(data_dir)

    # Dictionary to store the dataset
    dataset=dict()
    dataset['train'] = []
    dataset['validation'] = []
    dataset['test'] = []

    # Load the training data of CIFAR-100
    with open(data_dir + CIFAR_100_DIR + '/train', 'rb') as f:
        datadict = pickle.load(f,encoding='bytes')
    #print(datadict)
    _X = datadict[b'data']
    _Y = np.array(datadict[b'fine_labels'])
    _Y = dense_to_one_hot(_Y, num_classes=100)

    _X = np.array(_X, dtype=float) / 255.0
    _X = _X.reshape([-1, 3, 32, 32])
    _X = _X.transpose([0, 2, 3, 1])
    #_X = _X.reshape([-1, 3*32*32])
    x_train_mean = np.mean(_X, axis=0)
    x_train = _X#[:1000]
    y_train = _Y#[:1000]

    # Load the test batch of CIFAR-100
    with open(data_dir + CIFAR_100_DIR + '/test', 'rb') as f:
        datadict = pickle.load(f,encoding='bytes')

    _X = datadict[b'data']
    _Y = np.array(datadict[b'fine_labels'])
    _Y = dense_to_one_hot(_Y, num_classes=100)

    _X = np.array(_X, dtype=float) / 255.0
    _X = _X.reshape([-1, 3, 32, 32])
    _X = _X.transpose([0, 2, 3, 1])
    #_X = _X.reshape([-1,3*32*32])

    x_test = _X
    y_test = _Y

    # Normalize the test set
    x_train -= x_train_mean
    x_test -= x_train_mean

    dataset['train'].append(x_train)
    dataset['train'].append(y_train)

    dataset['test'].append(x_test)
    dataset['test'].append(y_test)

    return x_train,y_train,x_test,y_test


def split_cifar(x,y,x_,y_,cond,num_classes):

    def dense_to_one_hot(labels_dense, num_classes=100):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()%num_classes] = 1

        return labels_one_hot

    sets_list=[]
    maxlabels = np.argmax(y, 1)
    maxlabels = [True if x in cond else False for x in maxlabels]
    sets_list.append(DataSet(x[maxlabels,:], dense_to_one_hot( np.argmax( y[maxlabels],1),num_classes),
                                dtype=dtypes.uint8, reshape=False))

    maxlabels = np.argmax(y_, 1)
    maxlabels = [True if x in cond else False for x in maxlabels]
    sets_list.append(DataSet(x_[maxlabels,:], dense_to_one_hot( np.argmax( y_[maxlabels],1),num_classes),
                                dtype=dtypes.uint8, reshape=False))

    return base.Datasets(train=sets_list[0], validation=[], test=sets_list[1])

def train(task_list,task_labels,FLAGS):
    #return [[0.0 for i in range(20)] for j in range(20)]
    # Training
    # ==================================================
    g1 = tf.Graph()
    
    with g1.as_default():
        #optimizer=tf.train.MomentumOptimizer(0.001, momentum=0.9)
        if FLAGS.optim == 'SGD':
            optimizer=tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        elif FLAGS.optim == 'MOMENTUM':
            optimizer= tf.train.MomentumOptimizer(FLAGS.learning_rate)
        elif FLAGS.optim == 'ADAM':
            optimizer= tf.train.AdamOptimizer(FLAGS.learning_rate)

        if FLAGS.method=='RLL':
            from Model_RGO import RGO_Net
            Model = RGO_Net(FLAGS.arch,num_classes=FLAGS.num_class,dim = FLAGS.dim, seed_num=FLAGS.seed, optimizer=optimizer)
        elif FLAGS.method=='OCL':
            from Model_OCL import OCL_Net
            Model = OCL_Net(FLAGS.arch,num_classes=FLAGS.num_class,dim = FLAGS.dim, seed_num=FLAGS.seed, optimizer=optimizer)
        elif FLAGS.method=='SGD' or FLAGS.method=='MTL' or FLAGS.method=='STL':
            from Model_VAN import SGD_Net
            Model = SGD_Net(FLAGS.arch,num_classes=FLAGS.num_class,dim = FLAGS.dim, seed_num=FLAGS.seed, optimizer=optimizer)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if 'OCL' in FLAGS.method:

        with tf.Session(graph=g1, config=config) as sess1:
            # Initialize all variables
            init = [tf.global_variables_initializer()]#, tf.local_variables_initializer()]
            sess1.run(init)
            task_num = len(task_list)
            print('task num',task_num)
            accuracy_lists=[]#[[] for i in range(task_num)]
            train_accuracy_lists=[]
            loss_lists=[]
            total_accuracy_list=[]
            ops=[]
            
            for j in range(0, task_num):
                print("Training Task %d" % (j + 1))
                ops.append(Model.get_ops(j))
                # Update the parameters
                epoch = FLAGS.epoch
                batch_size = FLAGS.batch_size
                all_data = len(task_list[j].train.labels[:])
                all_step = all_data*epoch//batch_size
                print('all step:',all_step)
                if FLAGS.maxstep >0:
                    all_step = FLAGS.maxstep
                
                for current_step in range(all_step):
                    lamda = current_step/all_step
                    #current_step = current_step+1
                    batch_xs, batch_ys = task_list[j].train.next_batch(batch_size)
                    output_mask=np.zeros(FLAGS.num_class)
                    output_mask[task_labels[j]]=1.0

                    feed_dict = {
                        Model.input_x: batch_xs,
                        Model.input_y: batch_ys,
                        Model.output_mask:output_mask,
                        Model.train_phase:True
                    }
                    acc, loss,  aaa, = sess1.run(ops[j][0:3], feed_dict,)

                    Model.optimizer.update_theta(Model.trainable_vars)

                    if current_step % (all_step // epoch) == 0:
                        print("Train->>>Task: [{:d}/{:d}] Step: {:d}/{:d} Train: loss: {:.2f}, acc: {:.2f}  %"
                            .format(j+1, task_num,current_step*epoch // all_step+1, epoch, loss, acc * 100))
                print('Task{:d} is trained.'.format(j+1))



                accus=[]
                train_accus=[]
                losses = []
                for i_test in range(task_num):
                    output_mask=np.zeros(FLAGS.num_class)
                    output_mask[task_labels[i_test]]=1.0
                    # return None
                    all_data = len(task_list[i_test].train.labels[:])
                    all_step = all_data//batch_size
                    accuT,lossT=0.0,0.0
                    for index in range(all_step):
                        batch_xs, batch_ys = task_list[i_test].train.next_batch(batch_size)
                        feed_dict = {
                            Model.input_x: batch_xs,
                            Model.input_y: batch_ys,
                            Model.output_mask:output_mask,
                            Model.train_phase:False
                        }
                        if i_test<=j:
                            accu, loss = sess1.run(ops[i_test][0:2], feed_dict)
                        else:
                            accu, loss = 0.0 , 0.0
                        accuT += accu
                        lossT += loss
                    train_accus.append(accuT/all_step)

                    all_data = len(task_list[i_test].test.labels[:])
                    all_step = all_data//batch_size
                    accuT,lossT=0.0,0.0
                    for index in range(all_step):
                        batch_xs, batch_ys = task_list[i_test].test.next_batch(batch_size)
                        feed_dict = {
                            Model.input_x: batch_xs,
                            Model.input_y: batch_ys,
                            Model.output_mask:output_mask,
                            Model.train_phase:False
                        }
                        if i_test<=j:
                            accu, loss = sess1.run(ops[i_test][0:2], feed_dict)
                        else:
                            accu, loss = 0.0 , 0.0
                        accuT += accu
                        lossT += loss
                    accus.append(accuT/all_step)
                    losses.append(lossT/all_step)

                accuracy_lists.append(accus)
                train_accuracy_lists.append(train_accus)
                loss_lists.append(losses)
                total_accuracy_list.append(np.mean(accus[0:j+1]))

                #print("Test:->>>[{:d}/{:d}], Step: [{:d}/{:d}], acc: {:.2f} %".format(i_test + 1, task_num, current_step*epoch // all_step+1, epoch, accu * 100))
                print("Test:->>>accuracy of trainset:",100.0*np.array(train_accus))
                print("Test:->>>accuracy of tasks:",100.0*np.array(accus),"total_accuracy:",total_accuracy_list[-1]*100.0)
                
                print('updating P for Recursive Least Loss')
                update_batch_size = batch_size
                all_step = all_data//update_batch_size
                if FLAGS.maxstep >0:
                    all_step = FLAGS.maxstep*batch_size//update_batch_size
                for current_step in range(all_step):
                    #print(current_step,end=' ')
                    lamda = current_step/all_step
                    current_step = current_step+1
                    batch_xs, batch_ys = task_list[j].train.next_batch(update_batch_size)
                    output_mask=np.ones(FLAGS.num_class)
                    feed_dict = {
                        Model.input_x: batch_xs,
                        Model.input_y: batch_ys,
                        Model.output_mask:output_mask,
                        Model.train_phase:False
                    }
                    aaa, = sess1.run([Model.update], feed_dict,)
                        
                print('updated P for Recursive Least Loss')

            sess1.close()
    else:
        with tf.Session(graph=g1, config=config) as sess1:
            # Initialize all variables
            init = [tf.global_variables_initializer()]#, tf.local_variables_initializer()]
            sess1.run(init)
            
            task_num = len(task_list)
            accuracy_lists=[]#[[] for i in range(task_num)]
            train_accuracy_lists=[]
            loss_lists=[]
            total_accuracy_list=[]
            ops=[]
            for j in range(0, task_num):
                print("Training Task %d" % (j + 1))
                ops.append(Model.get_ops(j))
                # Update the parameters
                if FLAGS.method == 'STL':
                    init = [tf.global_variables_initializer()]#, tf.local_variables_initializer()]
                    sess1.run(init)
                epoch = FLAGS.epoch
                batch_size = FLAGS.batch_size
                all_data = len(task_list[j].train.labels[:])
                all_step = all_data*epoch//batch_size
                if FLAGS.maxstep >0:
                    all_step = FLAGS.maxstep
                if 'MTL' in FLAGS.method and j==0:
                    # merge all tasks into one task
                    # multiply the total steps as we only train task 0
                    #all_step *= task_num
                    print('multi task merging')
                    train_images = task_list[0].train.images
                    train_labels = task_list[0].train.labels
                    for jj in range(1,task_num):
                        train_images = np.concatenate((train_images, task_list[jj].train.images),axis=0)
                        train_labels = np.concatenate((train_labels, task_list[jj].train.labels),axis=0)
                    perm = np.arange(train_labels.shape[0])
                    np.random.shuffle(perm)
                    task_list[0] = base.Datasets(train=DataSet(train_images[perm,:], train_labels[perm],
                                            dtype=dtypes.uint8, reshape=False), validation=task_list[0].validation, test=task_list[0].test)
                    print('multi task merging end')
                
                for current_step in range(all_step):
                    lamda = current_step/all_step
                    #current_step = current_step+1
                    batch_xs, batch_ys = task_list[j].train.next_batch(batch_size)
                    output_mask=np.zeros(FLAGS.num_class)
                    output_mask[task_labels[j]]=1.0
                    if 'MTL' in FLAGS.method:
                        output_mask[:]=1.0
                        if j > 0 :
                            #skip tasks except the merged task 'task0'
                            break
                    # if FLAGS.method== 'RLL':
                    #     output_mask[:]=1.0
                    feed_dict = {
                        Model.input_x: batch_xs,
                        Model.input_y: batch_ys,
                        Model.output_mask:output_mask,
                        Model.train_phase:True
                    }
                    acc, loss,  aaa, = sess1.run(ops[j][0:3], feed_dict,)
                    if current_step % (all_step // epoch) == 0:
                        print("Train->>>Task: [{:d}/{:d}] Step: {:d}/{:d} Train: loss: {:.2f}, acc: {:.2f}  %"
                            .format(j+1, task_num,current_step*epoch // all_step+1, epoch, loss, acc * 100))
                print('Task{:d} is trained.'.format(j+1))
                # if current_step % (all_step // 4) == 0:
                #     print("Train->>>Task: [{:d}/{:d}] Step: {:d}/{:d} Train: loss: {:.2f}, acc: {:.2f}  %"
                #             .format(j+1, task_num,current_step*epoch // all_step+1, epoch, loss, acc * 100))
                # #print(current_step,all_step,FLAGS.evals)
                # if (current_step % (all_step//FLAGS.evals)) == 0 or (current_step==all_step-1) :
                accus=[]
                train_accus=[]
                losses = []
                for i_test in range(task_num):
                    output_mask=np.zeros(FLAGS.num_class)
                    output_mask[task_labels[i_test]]=1.0
                    all_data = len(task_list[i_test].train.labels[:])
                    all_step = all_data//batch_size
                    accuT,lossT=0.0,0.0
                    for index in range(all_step):
                        batch_xs, batch_ys = task_list[i_test].train.next_batch(batch_size)
                        feed_dict = {
                            Model.input_x: batch_xs,
                            Model.input_y: batch_ys,
                            Model.output_mask:output_mask,
                            Model.train_phase:False
                        }
                        if i_test<=j:
                            accu, loss = sess1.run(ops[i_test][0:2], feed_dict)
                        else:
                            accu, loss = 0.0 , 0.0
                        accuT += accu
                        lossT += loss
                    train_accus.append(accuT/all_step)

                    all_data = len(task_list[i_test].test.labels[:])
                    all_step = all_data//batch_size
                    accuT,lossT=0.0,0.0
                    for index in range(all_step):
                        batch_xs, batch_ys = task_list[i_test].test.next_batch(batch_size)
                        feed_dict = {
                            Model.input_x: batch_xs,
                            Model.input_y: batch_ys,
                            Model.output_mask:output_mask,
                            Model.train_phase:False
                        }
                        if i_test<=j:
                            accu, loss = sess1.run(ops[i_test][0:2], feed_dict)
                        else:
                            accu, loss = 0.0 , 0.0
                        accuT += accu
                        lossT += loss
                    accus.append(accuT/all_step)
                    losses.append(lossT/all_step)

                if FLAGS.method=='STL' and j>0:
                    accus[0:j]=accuracy_lists[-1][0:j]
                    train_accus[0:j]=train_accuracy_lists[-1][0:j]
                accuracy_lists.append(accus)
                train_accuracy_lists.append(train_accus)
                loss_lists.append(losses)
                total_accuracy_list.append(np.mean(accus[0:j+1]))

                #print("Test:->>>[{:d}/{:d}], Step: [{:d}/{:d}], acc: {:.2f} %".format(i_test + 1, task_num, current_step*epoch // all_step+1, epoch, accu * 100))
                print("Test:->>>accuracy of trainset:",100.0*np.array(train_accus))
                print("Test:->>>accuracy of tasks:",100.0*np.array(accus),"total_accuracy:",total_accuracy_list[-1]*100.0)
                
                if FLAGS.method=='RLL':
                    print('updating P for Recursive Least Loss')
                    update_batch_size = batch_size
                    all_step = all_data//update_batch_size
                    if FLAGS.maxstep >0:
                        all_step = FLAGS.maxstep*batch_size//update_batch_size
                    for current_step in range(all_step):
                        #print(current_step,end=' ')
                        lamda = current_step/all_step
                        current_step = current_step+1
                        batch_xs, batch_ys = task_list[j].train.next_batch(update_batch_size)
                        output_mask=np.ones(FLAGS.num_class)
                        feed_dict = {
                            Model.input_x: batch_xs,
                            Model.input_y: batch_ys,
                            Model.output_mask:output_mask,
                            Model.train_phase:False
                        }
                        aaa, = sess1.run([Model.update], feed_dict,)
                        
                    print('updated P for Recursive Least Loss')
                
            # dict_plot={'method':FLAGS.method,'accu':accuracy_lists}
            # with open("result_cf100_"+FLAGS.method+'1','wb') as f:
            #     pickle.dump(dict_plot,f)
            sess1.close()
    del g1
    return np.array(accuracy_lists),np.array(loss_lists),np.array(train_accuracy_lists)

def main(_):

    FLAGS = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpuid)
    if FLAGS.dataset == 'CIFAR':
        FLAGS.num_class = 100
        FLAGS.dim = [32,32,3]
        FLAGS.sequence_length=np.prod(FLAGS.dim)
        x,y,x_,y_=_get_cifar('data/CIFAR_data')
    elif 'MNIST' in FLAGS.dataset:
        FLAGS.num_class = 10
        FLAGS.dim = [28*28]
        FLAGS.sequence_length=np.prod(FLAGS.dim)
        mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
    elif FLAGS.dataset == 'miniImageNet':
        FLAGS.num_class = 100
        FLAGS.dim = [84,84,3]
        FLAGS.sequence_length=np.prod(FLAGS.dim)
    np.random.seed(FLAGS.seed)
    num_classes=FLAGS.num_class
    print("result_"+FLAGS.dataset+"_"+FLAGS.method+"_"+FLAGS.arch+"_lr_"+str(FLAGS.learning_rate)+'_epochs_'+str(FLAGS.epoch)+'_batchsize_'+str(FLAGS.batch_size)+'_runs_'+str(FLAGS.num_runs))
    runs = []
    runs_train=[]
    runs_loss=[]
    runs_label=[]
    for runid in range(FLAGS.num_runs):
        print('Runid: '+str(runid))
        task_labels = np.arange(num_classes)
        np.random.shuffle(task_labels)
        runs_label.append(task_labels)
        print(task_labels)
        if FLAGS.dataset == 'CIFAR':
            task_labels = [task_labels[i*num_classes//FLAGS.num_task : (i+1)*num_classes//FLAGS.num_task] for i in range(FLAGS.num_task)]
            task_list=[]
            for i in range(FLAGS.num_task):
                task_list.append(split_cifar(x,y,x_,y_, task_labels[i], num_classes ))
        elif FLAGS.dataset == 'p-MNIST':
            task_list=[]
            task_labels = [np.arange(num_classes)]*FLAGS.num_task
            for i in range(FLAGS.num_task):
                task_list.append(shuffle_mnist(mnist,i+runid*FLAGS.num_task))
        elif FLAGS.dataset == 'r-MNIST':
            task_list=[]
            task_labels = [np.arange(num_classes)]*FLAGS.num_task
            for tt in range(FLAGS.num_task):
                random.seed(a=tt*1234)
                min_angle, max_angle=0,180
                min_rot = 1.0 * tt / FLAGS.num_task * (max_angle - min_angle) + min_angle
                max_rot = 1.0 * (tt + 1) / FLAGS.num_task * (max_angle - min_angle) + min_angle
                task_rotation = random.random() * (max_rot - min_rot) + min_rot
                task_list.append(rotate_mnist(mnist,task_rotation))
        elif FLAGS.dataset == 'miniImageNet':
            task_labels = [task_labels[i*num_classes//FLAGS.num_task : (i+1)*num_classes//FLAGS.num_task] for i in range(FLAGS.num_task)]
            DATA_FILE = '/data/miniImageNet_Dataset/miniImageNet_full.pickle'
            task_list = construct_split_miniImagenet(task_labels,DATA_FILE)
        # print(FLAGS)
        acculist,losslist,train_acculist = train(task_list,task_labels,FLAGS)
        runs.append(np.array(acculist))
        runs_train.append(np.array(train_acculist))
        runs_loss.append(np.array(losslist))
    runs = np.array(runs)
    runs_train = np.array(runs_train)
    runs_loss = np.array(runs_loss)
    runs_label = np.array(runs_label)
    dict_plot={'method':FLAGS.method,'mean':runs,'loss':runs_loss,'task':runs_label,'train':runs_train}
    with open(FLAGS.log_dir+"result_"+FLAGS.dataset+"_"+FLAGS.method+"_"+FLAGS.arch+"_lr_"+str(FLAGS.learning_rate)+'_epochs_'+str(FLAGS.epoch)+'_batchsize_'+str(FLAGS.batch_size)+'_runs_'+str(FLAGS.num_runs)+'.pickle','wb') as f:
        pickle.dump(dict_plot,f)

if __name__ == '__main__':
    tf.app.run()
