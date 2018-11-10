########## load packages ##########
import tensorflow as tf
import numpy as np

########## set net hyperparameters ##########
learning_rate = 0.0001

epochs = 20

steps = 1000

batch_size_train = 256
batch_size_test = 100

display_step = 20

scale = 0.1

########## set net parameters ##########
#### img shape:299*299 ####
height = 299
width = 299

#### 102 classes flowers ####
n_classes = 102

# Handle Dimension Ordering for different backends
'''
img_input_shape=(224, 224, 3)
concat_axis = 3

img_input_shape=(3, 224, 224)
concat_axis=1
'''
global concat_axis

concat_axis = 3

########## placeholder ##########
x = tf.placeholder(tf.float32, [None, height, width, 3])
y = tf.placeholder(tf.float32, [None, n_classes])


############### get flower data train ###############
def flower_batch(filename, batch_size):
    '''
    filename: TFRecord路径
    '''

    ########### 根据文件名生成一个队列 ############
    filename_queue = tf.train.string_input_producer([filename])

    ########### 生成 TFRecord 读取器 ############
    reader = tf.TFRecordReader()

    ########### 返回文件名和文件 ############
    _, serialized_example = reader.read(filename_queue)

    ########### 取出example里的features #############
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64)})

    ########### 将序列化的img转为uint8的tensor #############
    img = tf.decode_raw(features['img'], tf.uint8)

    ########### 将label转为int32的tensor #############
    label = tf.cast(features['label'], tf.int32)

    ########### 将图片调整成正确的尺寸 ###########
    img = tf.reshape(img, [height, width, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    ########### 批量输出图片, 使用shuffle_batch可以有效地随机从训练数据中抽出batch_size个数据样本 ###########
    ##### shuffle batch之前，必须提前定义影像的size，size不可以是tensor，必须是明确的数字 ######
    ##### num_threads 表示可以选择用几个线程同时读取 #####
    ##### min_after_dequeue 表示读取一次之后队列至少需要剩下的样例数目 #####
    ##### capacity 表示队列的容量 #####
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity=100, num_threads=2,
                                                    min_after_dequeue=10)

    return img_batch, label_batch


############### label one hot ###############
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


##################### build net model ##########################
################ conv block ###################
def conv_block(input, filter, kernel_size, stride, padding, is_activation=True):
    '''
    :param input: input image
    :param filter: 卷积通道数
    :param kernel_size: 卷积核参数
    :param stride: 步长
    :param padding: 卷积方式

    :return: 卷积 ——> BN ——> relu ——> out
    '''

    ####### conv ########
    conv = tf.layers.conv2d(input, filters=filter, kernel_size=kernel_size, strides=stride, padding=padding)

    ####### BN ########
    conv = tf.layers.batch_normalization(conv)

    if is_activation:
        ####### relu ########
        conv = tf.nn.relu(conv)

    return conv


################ stem ###################
def stem(input):
    '''
    :param input: input image

    :return:
            3个普通卷积 (conv 3x3 filter=32 stride=2 VALID;
                       conv 3x3 filter=32 stride=1 VALID;
                       conv 3x3 filter=64 stride=2 SAME)

            ——> 两个分支做Filter concat (branch1_1: maxpool 3x3 stride=2 VALID;
                                       branch1_2: conv 3x3 filter=96 stride=2 VALID)

            ——> 两个分支做Filter concat (branch2_1: conv 1x1 filter=64 stride=1 SAME, conv 3x3 filter=96 stride=1 VALID;

                                       branch2_2: conv 1x1 filter=64 stride=1 SAME, conv 7x1 filter=64 stride=1 SAME,
                                                  conv 1x7 filter=64 stride=1 SAME, conv 3x3 filter=96 stride=1 VALID)

            ——> 两个分支做Filter concat (branch3_1: conv 3x3 filter=192 stride=2 VALID;
                                       branch3_2: maxpool 3x3 stride=2 VALID)

            ——> out
    '''
    ######## 3个普通卷积 ########
    #### conv 3x3 filter=32 stride=2 VALID ####
    conv1 = conv_block(input, filter=32, kernel_size=3, stride=2, padding='VALID')

    #### conv 3x3 filter=32 stride=1 VALID ####
    conv2 = conv_block(conv1, filter=32, kernel_size=3, stride=1, padding='VALID')

    #### conv 3x3 filter=64 stride=2 SAME ####
    conv3 = conv_block(conv2, filter=64, kernel_size=3, stride=1, padding='SAME')


    ######## 两个分支 ########
    #### maxpool 3x3 stride=2 VALID ####
    branch1_1 = tf.layers.max_pooling2d(conv3, pool_size=(3, 3), strides=2, padding='VALID')

    #### conv 3x3 filter=96 stride=2 VALID ####
    branch1_2 = conv_block(conv3, filter=96, kernel_size=3, stride=2, padding='VALID')

    #### Filter concat ####
    branch1_concat = tf.concat([branch1_1, branch1_2], concat_axis)


    ######## 两个分支 ########
    #### conv 1x1 filter=64 stride=1 SAME ####
    branch2_1 = conv_block(branch1_concat, filter=64, kernel_size=1, stride=1, padding='SAME')
    #### conv 3x3 filter=96 stride=1 VALID ####
    branch2_1 = conv_block(branch2_1, filter=96, kernel_size=3, stride=1, padding='VALID')

    #### conv 1x1 filter=64 stride=1 SAME ####
    branch2_2 = conv_block(branch1_concat, filter=64, kernel_size=1, stride=1, padding='SAME')
    #### conv 7x1 filter=64 stride=1 SAME ####
    branch2_2 = conv_block(branch2_2, filter=64, kernel_size=[7, 1], stride=1, padding='SAME')
    #### conv 1x7 filter=64 stride=1 SAME ####
    branch2_2 = conv_block(branch2_2, filter=64, kernel_size=[1, 7], stride=1, padding='SAME')
    #### conv 3x3 filter=96 stride=1 VALID ####
    branch2_2 = conv_block(branch2_2, filter=96, kernel_size=3, stride=1, padding='VALID')

    #### Filter concat ####
    branch2_concat = tf.concat([branch2_1, branch2_2], concat_axis)


    ######## 两个分支 ########
    #### conv 3x3 filter=192 stride=2 VALID ####
    branch3_1 = conv_block(branch2_concat, filter=192, kernel_size=3, stride=2, padding='VALID')

    #### maxpool 3x3 stride=2 VALID ####
    branch3_2 = tf.layers.max_pooling2d(branch2_concat, pool_size=(3, 3), strides=2, padding='VALID')

    #### Filter concat ####
    branch3_concat = tf.concat([branch3_1, branch3_2], concat_axis)

    return branch3_concat


################ Inception_A ################
def Inception_A(input):
    '''

    :param input: input image

    :return: ——> input +

            {  3个分支做Filter concat
                                    branch1: conv 1x1 filter=32 stride=1 SAME;

                                    branch2: conv 1x1 filter=32 stride=1 SAME;
                                             conv 3x3 filter=32 stride=1 SAME;

                                    branch3: conv 1x1 filter=32 stride=1 SAME,
                                             conv 3x3 filter=48 stride=1 SAME,
                                             conv 3x3 filter=64 stride=1 SAME;

            ——> conv 1x1 filter=384 stride=1 SAME 激活函数为线性;  }

    '''

    ############### init ###############
    init = input


    ############### branch ###############
    ######## branch1 ########
    #### conv 1x1 filter=32 stride=1 SAME ####
    branch1 = conv_block(input, filter=32, kernel_size=1, stride=1, padding='SAME')

    ######## branch2 ########
    #### conv 1x1 filter=32 stride=1 SAME ####
    branch2 = conv_block(input, filter=32, kernel_size=1, stride=1, padding='SAME')
    #### conv 3x3 filter=32 stride=1 SAME ####
    branch2 = conv_block(branch2, filter=32, kernel_size=3, stride=1, padding='SAME')

    ######## branch3 ########
    #### conv 1x1 filter=32 stride=1 SAME ####
    branch3 = conv_block(input, filter=32, kernel_size=1, stride=1, padding='SAME')
    #### conv 3x3 filter=48 stride=1 SAME ####
    branch3 = conv_block(branch3, filter=48, kernel_size=3, stride=1, padding='SAME')
    #### conv 3x3 filter=64 stride=1 SAME ####
    branch3 = conv_block(branch3, filter=64, kernel_size=3, stride=1, padding='SAME')

    ######## Filter concat ########
    branch = tf.concat([branch1, branch2, branch3], concat_axis)

    ######## conv 1x1 filter=384 stride=1 SAME 激活函数为线性 ########
    branch = conv_block(branch, filter=384, kernel_size=1, stride=1, padding='SAME', is_activation=False)


    ############### add init and branch ###############
    out = tf.add(init, branch * scale)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


################ Reduction_A ################
def Reduction_A(input):
    '''
    :param input: input image

    :return: 3个分支做Filter concat
                                  branch1: maxpool 3x3 stride=2 VALID;

                                  branch2: conv 3x3 filter=384 stride=2 VALID;

                                  branch3: conv 1x1 filter=256 stride=1 SAME,
                                           conv 3x3 filter=256 stride=1 SAME,
                                           conv 3x3 filter=384 stride=2 SAME;
    '''
    ######## branch1 ########
    #### maxpool 3x3 stride=2 VALID ####
    branch1 = tf.layers.max_pooling2d(input, pool_size=(3, 3), strides=2, padding='VALID')


    ######## branch2 ########
    #### conv 3x3 filter=384 stride=2 VALID ####
    branch2 = conv_block(input, filter=384, kernel_size=3, stride=2, padding='VALID')


    ######## branch3 ########
    #### conv 1x1 filter=256 stride=1 SAME ####
    branch3 = conv_block(input, filter=256, kernel_size=1, stride=1, padding='SAME')
    #### conv 3x3 filter=256 stride=1 SAME ####
    branch3 = conv_block(branch3, filter=256, kernel_size=3, stride=1, padding='SAME')
    #### conv 3x3 filter=384 stride=2 SAME ####
    branch3 = conv_block(branch3, filter=384, kernel_size=3, stride=2, padding='VALID')


    ######## Filter concat ########
    out = tf.concat([branch1, branch2, branch3], concat_axis)

    return out

################ Inception_B ################
def Inception_B(input):
    '''
    :param input: input image
    :return: ——> input +

            {  2个分支做Filter concat
                                    branch1: conv 1x1 filter=192 stride=1 SAME;

                                    branch2: conv 1x1 filter=128 stride=1 SAME,
                                             conv 1x7 filter=160 stride=1 SAME,
                                             conv 7x1 filter=192 stride=1 SAME;

            ——> conv 1x1 filter=1152 stride=1 SAME 激活函数为线性;  }
    '''
    ############### init ###############
    init = input


    ############### branch ###############
    ######## branch1 ########
    #### conv 1x1 filter=192 stride=1 SAME ####
    branch1 = conv_block(input, filter=192, kernel_size=1, stride=1, padding='SAME')

    ######## branch2 ########
    #### conv 1x1 filter=128 stride=1 SAME ####
    branch2 = conv_block(input, filter=128, kernel_size=1, stride=1, padding='SAME')
    #### conv 1x7 filter=160 stride=1 SAME ####
    branch2 = conv_block(branch2, filter=160, kernel_size=[1, 7], stride=1, padding='SAME')
    #### conv 7x1 filter=192 stride=1 SAME ####
    branch2 = conv_block(branch2, filter=192, kernel_size=[7, 1], stride=1, padding='SAME')

    ######## Filter concat ########
    branch = tf.concat([branch1, branch2], concat_axis)

    ######## conv 1x1 filter=1152 stride=1 SAME 激活函数为线性 ########
    branch = conv_block(branch, filter=1152, kernel_size=1, stride=1, padding='SAME', is_activation=False)


    ############### add init and branch ###############
    out = tf.add(init, branch * scale)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


################## Reduction_B #################
def Reduction_B(input):
    '''
    :param input: input image

    :return: 4个分支做Filter concat
                                  branch1: maxpool 3x3 stride=2 VALID;

                                  branch2: conv 1x1 filter=256 stride=1 SAME,
                                           conv 3x3 filter=384 stride=2 VALID;

                                  branch3: conv 1x1 filter=256 stride=1 SAME,
                                           conv 3x3 filter=288 stride=2 VALID;

                                  branch4: conv 1x1 filter=256 stride=1 SAME,
                                           conv 3x3 filter=288 stride=1 SAME,
                                           conv 3x3 filter=320 stride=2 VALID;

    '''

    ######## branch1 ########
    #### maxpool 3x3 stride=2 VALID ####
    branch1 = tf.layers.max_pooling2d(input, pool_size=(3, 3), strides=2, padding='VALID')


    ######## branch2 ########
    #### conv 1x1 filter=256 stride=1 SAME ####
    branch2 = conv_block(input, filter=256, kernel_size=1, stride=1, padding='SAME')
    #### conv 3x3 filter=384 stride=2 VALID ####
    branch2 = conv_block(branch2, filter=384, kernel_size=3, stride=2, padding='VALID')


    ######## branch3 ########
    #### conv 1x1 filter=256 stride=1 SAME ####
    branch3 = conv_block(input, filter=256, kernel_size=1, stride=1, padding='SAME')
    #### conv 3x3 filter=288 stride=2 VALID ####
    branch3 = conv_block(branch3, filter=288, kernel_size=3, stride=2, padding='VALID')

    ######## branch4 ########
    #### conv 1x1 filter=256 stride=1 SAME ####
    branch4 = conv_block(input, filter=256, kernel_size=1, stride=1, padding='SAME')
    #### conv 3x3 filter=288 stride=1 SAME ####
    branch4 = conv_block(branch4, filter=288, kernel_size=3, stride=1, padding='SAME')
    #### conv 3x3 filter=320 stride=2 VALID ####
    branch4 = conv_block(branch4, filter=320, kernel_size=3, stride=2, padding='VALID')


    ######## Filter concat ########
    out = tf.concat([branch1, branch2, branch3, branch4], concat_axis)

    return out


################### Inception_C ####################
def Inception_C(input):
    '''

        :param input: input image

        :return: ——> input +

            {  2个分支做Filter concat
                                    branch1: conv 1x1 filter=192 stride=1 SAME;

                                    branch2: conv 1x1 filter=192 stride=1 SAME,
                                             conv 1x3 filter=224 stride=1 SAME,
                                             conv 3x1 filter=256 stride=1 SAME;

            ——> conv 1x1 filter=2144 stride=1 SAME 激活函数为线性;  }

        '''

    ############### init ###############
    init = input


    ######## branch1 ########
    #### conv 1x1 filter=192 stride=1 SAME ####
    branch1 = conv_block(input, filter=192, kernel_size=1, stride=1, padding='SAME')


    ######## branch2 ########
    #### conv 1x1 filter=192 stride=1 SAME ####
    branch2 = conv_block(input, filter=192, kernel_size=1, stride=1, padding='SAME')
    #### conv 1x3 filter=224 stride=1 SAME ####
    branch2 = conv_block(branch2, filter=224, kernel_size=[1, 3], stride=1, padding='SAME')
    #### conv 3x1 filter=256 stride=1 SAME ####
    branch2 = conv_block(branch2, filter=256, kernel_size=[3, 1], stride=1, padding='SAME')

    ######## Filter concat ########
    branch = tf.concat([branch1, branch2], concat_axis)

    ######## conv 1x1 filter=2144 stride=1 SAME 激活函数为线性 ########
    branch = conv_block(branch, filter=2144, kernel_size=1, stride=1, padding='SAME', is_activation=False)

    ############### add init and branch ###############
    out = tf.add(init, branch * scale)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


################### GoogleNet Inception resnet_v2 ##################
def Inception_resnet_v2(x, n_classes):
    '''
    x: input image
    n_classes: 类别，可根据数据集调整

    return:
           out: 输出
    '''

    ####### reshape input picture ########
    x = tf.reshape(x, shape=[-1, height, width, 3])

    ####### stem ########
    x = stem(x)

    ####### 5 x Inception_A ########
    for _ in range(5):
        x = Inception_A(x)

    ####### Reduction_A ########
    x = Reduction_A(x)

    ####### 10 x Inception_B ########
    for _ in range(10):
        x = Inception_B(x)

    ####### Reduction_B ########
    x = Reduction_B(x)

    ####### 5 x Inception_C ########
    for _ in range(5):
        x = Inception_C(x)

    ####### 8x8 avgpool ########
    x = tf.layers.average_pooling2d(x, pool_size=(8, 8), strides=1, padding='VALID')

    ###### flatten 影像展平 ########
    x = tf.reshape(x, (-1, 1 * 1 * 2144))

    ####### dropout (keep 0.8) ########
    x = tf.nn.dropout(x, 0.8)

    #### out ####
    out = tf.layers.dense(x, n_classes)

    return out


#################### define model, loss and optimizer ###################
#### model pred 影像输出结果 ####
out = Inception_resnet_v2(x, n_classes)

#### loss 损失计算 ####
## 阻断label的梯度流 ##
y = tf.stop_gradient(y)

## 主要输出结果 ##
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(out), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


##################### train and evaluate model ##########################
train_file = "flower_train_299.tfrecords"
x_train, y_train = flower_batch(train_file, batch_size_train)


##################### sess ###################
with tf.Session() as sess:
    ########## initialize variables ##########
    init = tf.global_variables_initializer()
    sess.run(init)

    ########## 启动队列线程 ##########
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #### epoch 世代循环 ####
    for epoch in range(epochs):

        #### 迭代取batch size ####
        for step in range(steps):

            #### train data ####
            x_train_batch, y_train_batch = sess.run([x_train, y_train])

            #### label one hot ####
            y_train_batch = np.reshape(y_train_batch, [batch_size_train, 1])
            y_train_batch = dense_to_one_hot(y_train_batch, n_classes)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: x_train_batch, y: y_train_batch})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: x_train_batch, y: y_train_batch})
                print("Epoch " + str(epoch) + ", Step " + str(step) + \
                  ", Training Loss=" + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc))

    ########## 关闭队列线程 ##########
    coord.request_stop()
    coord.join(threads)

########## 关闭sess ##########
sess.close()
del sess


print("Optimizer Finished!")


##################### evaluate model ##########################
test_file = "flower_test_299.tfrecords"
x_test, y_test = flower_batch(test_file, batch_size_test)

##### test accuracy #####
with tf.Session() as sess:

    ########## initialize variables ##########
    init = tf.global_variables_initializer()
    sess.run(init)

    ########## 启动队列线程 ##########
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ########## test ##########
    for _ in range(100):

        ########## get test data ##########
        x_test_batch, y_test_batch = sess.run([x_test, y_test])

        ########## test accuracy ##########
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: x_test_batch, y: y_test_batch}))