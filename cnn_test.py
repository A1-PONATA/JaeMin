from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import time
from keras.utils import np_utils
import os
import tensorflow as tf
import random



def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img,(64, 64), cv2.INTER_LINEAR)
    return resized

def load_train():
    X_train=[]
    X_train_id=[]
    y_train=[]
    start_time = time.time()

    print('Read train images')
    folders = ['11_000', '11_001', '11_002']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('/home','pirl', 'Downloads', 'testData',  fld, '11_000.jpg')
        print(path)

        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    # for fld in folders:
    #     index = folders.index(fld)
    #     print('Load folder {} (Index: {})'.format(fld, index))
    #     path = os.path.join('.', 'Downloads', 'testData', 'fld', '*.jpg')
    #     files = glob.glob(path)
    #     for fl in files:
    #         flbase = os.path.basename(fl)
    #         img = get_im_cv2(fl)
    #         X_train.append(img)
    #         X_train_id.append(flbase)
    #         y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('/home','pirl', ' Downloads', 'testData', '11_000','11_000.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    path = os.path.join('/home','pirl', ' Downloads', 'testData', '11_001','*.jpg')
    files = sorted(glob.glob(path))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test,X_test_id

def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    print(train_data.shape)
    train_data = train_data.transpose((0, 2,3, 1))
    train_data = train_data.transpose((0, 1,3, 2))
    print(train_data.shape)

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data/255
    train_target = np_utils.to_categorical(train_target,3)
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id

def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((2,0,1))
    # test_data = test_data.transpose((0, 2,3,1))
    # test_data = test_data.transpose((0,1,3,2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

train_data, train_target, train_id = read_and_normalize_train_data()

list1_shuf = []
list2_shuf = []
index_shuf = range(len(train_data))
random.shuffle(index_shuf)
for i in index_shuf:
    list1_shuf.append(train_data[i,:,:,:])
    list2_shuf.append(train_target[i,])

list1_shuf = np.array(list1_shuf, dtype=np.uint8)
list2_shuf = np.array(list2_shuf, dtype=np.uint8)

channel_in = 3
channel_out = 64
channel_out1 = 128

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, stride, stride,1], strides=[1, stride, stride, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, stride=2)

    conv2a = conv2d(x, weights['wc2'], biases['bc2'])
    conv2a = maxpool2d(conv2a, stride=2)
    conv2 = conv2d(conv2a, weights['wc3'], biases['bc3'])
    conv2 = maxpool2d(conv2, stride=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

start_time = time.time()
learning_rate = 0.01
epochs = 200
batch_size = 128
num_batches = list1_shuf.shape[0]/128
input_height = 64
input_width = 64
n_classes = 3
dropout = 0.5
display_step = 1
filter_height = 3
filter_width = 3
depth_in = 3
depth_out1 = 64
depth_out2 = 128
depth_out3 = 256

x = tf.placeholder(tf.float32, [None, input_height, input_width, depth_in])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'wc1' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_in, depth_out1])),
    'wc2' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_in, depth_out2])),
    'wc3' : tf.Variable(tf.random_normal([filter_height, filter_width, depth_in, depth_out3])),
    'wd1' : tf.Variable(tf.random_normal([int(input_height/8)*int(input_width/8)*256, 512])),
    'wd2' : tf.Variable(tf.random_normal([512, 512])),
    'out' : tf.Variable(tf.random_normal([512, n_classes]))
}
biases = {
    'bc1' : tf.Variable(tf.random_normal([64])),
    'bc2' : tf.Variable(tf.random_normal([128])),
    'bc3' : tf.Variable(tf.random_normal([256])),
    'bd1' : tf.Variable(tf.random_normal([512])),
    'bd2' : tf.Variable(tf.random_normal([512])),
    'out' : tf.Variable(tf.random_normal([n_classes])),
}
pred = conv_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

corret_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(corret_pred, tf.float32))

inin = tf.global_variables_initializer()

start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):
            batch_x, batch_y = list1_shuf[i*(batch_size):(i+1)*(batch_size)], list2_shuf[i*(batch_size):(i+1)*(batch_size)]
            sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y, keep_prob:dropout})
            loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.})
            if epochs % display_step == 0:
                print('Epoch:', '%4d' % (i+1), 'cost=', '{:.9f}'.format(loss), 'Training accuracy','{:.5f'.format(acc))

    print('Optimization Completed')
end_time = time.time()
print('Total processing time : ', end_time - start_time)
