########## load packages ##########
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

##################### load data ##########################
mnist = input_data.read_data_sets("mnist_sets", one_hot=True)

########## set net hyperparameters ##########
learning_rate = 0.001

epochs = 20
batch_size_train = 128
batch_size_test = 100

display_step = 20

########## set net parameters ##########
#### img shape:28*28 ####
n_input = 784

#### 0-9 digits ####
n_classes = 10

#### dropout probability
dropout = 0.6

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
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


##################### build net model ##########################
def inception(input, filters, strides, name="None"):
    '''
    input: input image
    filters:
            # 1x1:   f1,
            # 3x3 reduce:  f3_r,  # 3x3:   f3,
            # 5x5 reduce:  f5_r,  # 5x5:   f5,
            pool proj:  proj
    '''

    ######## filters ########
    f1, f3_r, f3, f5_r, f5, proj = filters

    ######## conv # 1x1 ########
    if f1 > 0:
        #### conv ####
        conv1 = tf.layers.conv2d(input, filters=f1, kernel_size=1, strides=strides, padding='SAME')
        #### BN ####
        conv1 = tf.layers.batch_normalization(conv1)
        #### relu ####
        conv1 = tf.nn.relu(conv1)

    ######## conv # 3x3 ########
    #### conv # 3x3 reduce ####
    conv3r = tf.layers.conv2d(input, filters=f3_r, kernel_size=1, strides=strides, padding='SAME')
    #### BN ####
    conv3r = tf.layers.batch_normalization(conv3r)
    #### relu ####
    conv3r = tf.nn.relu(conv3r)


    #### conv # 3x3 ####
    conv3 = tf.layers.conv2d(conv3r, filters=f3, kernel_size=3, strides=1, padding='SAME')
    #### BN ####
    conv3 = tf.layers.batch_normalization(conv3)
    #### relu ####
    conv3 = tf.nn.relu(conv3)


    ######## conv # 5x5 ########
    #### conv # 5x5 reduce ####
    conv5r = tf.layers.conv2d(input, filters=f5_r, kernel_size=1, strides=strides, padding='SAME')
    #### BN ####
    conv5r = tf.layers.batch_normalization(conv5r)
    #### relu ####
    conv5r = tf.nn.relu(conv5r)


    #### conv # 5x5 ####
    conv5 = tf.layers.conv2d(conv5r, filters=f5, kernel_size=3, strides=1, padding='SAME')
    #### BN ####
    conv5 = tf.layers.batch_normalization(conv5)
    #### relu ####
    conv5 = tf.nn.relu(conv5)


    ######## pool proj ########
    if name == "pass through":
        #### pool ####
        pool1 = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, strides, strides, 1], padding='SAME')

        ######## 连接 ########
        out = tf.concat([conv3, conv5, pool1], concat_axis)

    elif name == "max pool":
        #### pool ####
        pool1 = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, strides, strides, 1], padding='SAME')

        #### proj ####
        convproj = tf.layers.conv2d(pool1, filters=proj, kernel_size=1, strides=1, padding='SAME')
        convproj = tf.layers.batch_normalization(convproj)
        convproj = tf.nn.relu(convproj)

        ######## 连接 ########
        out = tf.concat([conv1, conv3, conv5, convproj], concat_axis)

    else:
        #### pool ####
        pool1 = tf.nn.avg_pool(input, ksize=[1, 3, 3, 1], strides=[1, strides, strides, 1], padding='SAME')

        #### proj ####
        convproj = tf.layers.conv2d(pool1, filters=proj, kernel_size=1, strides=1, padding='SAME')
        convproj = tf.layers.batch_normalization(convproj)
        convproj = tf.nn.relu(convproj)

        ######## 连接 ########
        out = tf.concat([conv1, conv3, conv5, convproj], concat_axis)

    return out


######### GoogleNet Inception v1 ##########
def Inception_v2(x, n_classes):
    '''
    x: input image
    n_classes: 类别，可根据数据集调整

    return:
           out:   主要输出
           out_1: 辅助分类器1输出
           out_2: 辅助分类器2输出

    '''

    ####### reshape input picture ########
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    ####### first conv ########
    #### conv 1 ####
    conv1 = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='SAME')
    #### BN ####
    conv1 = tf.layers.batch_normalization(conv1)
    #### relu ####
    conv1 = tf.nn.relu(conv1)

    ####### max pool ########
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### second conv ########
    #### conv 2_1 ####
    conv2 = tf.layers.conv2d(pool1, filters=192, kernel_size=1, strides=1, padding='VALID')
    #### BN ####
    conv2 = tf.layers.batch_normalization(conv2)
    #### relu ####
    conv2 = tf.nn.relu(conv2)

    #### conv 2_2 ####
    conv2 = tf.layers.conv2d(conv2, filters=192, kernel_size=3, strides=1, padding='SAME')
    #### BN ####
    conv2 = tf.layers.batch_normalization(conv2)
    #### relu ####
    conv2 = tf.nn.relu(conv2)

    ####### max pool ########
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### inception 3a ########
    ince3a = inception(pool2, [64, 64, 64, 64, 96, 32], strides=1, name="None")

    ####### inception 3b ########
    ince3b = inception(ince3a, [64, 64, 96, 64, 96, 64], strides=1, name="None")

    ####### inception 3c ########
    ince3c = inception(ince3b, [0, 128, 160, 64, 96, 0], strides=1, name="pass through")

    ####### inception 4a ########
    ince4a = inception(ince3c, [224, 64, 96, 96, 128, 128], strides=2, name="None")

    ####### inception 4b ########
    ince4b = inception(ince4a, [192, 96, 128, 96, 128, 128], strides=1, name="None")

    ####### inception 4c ########
    ince4c = inception(ince4b, [160, 128, 160, 128, 160, 128], strides=1, name="None")

    ####### inception 4d ########
    ince4d = inception(ince4c, [96, 128, 192, 160, 192, 128], strides=1, name="None")

    ####### inception 4e ########
    ince4e = inception(ince4d, [0, 128, 192, 192, 256, 0], strides=1, name="pass through")

    ####### inception 5a ########
    ince5a = inception(ince4e, [352, 192, 320, 160, 224, 128], strides=2, name="None")

    ####### inception 5b ########
    ince5b = inception(ince5a, [352, 192, 320, 192, 224, 128], strides=1, name="max pool")

    ####### average pool ########
    pool3 = tf.nn.avg_pool(ince5b, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    ###### flatten 影像展平 ########
    flatten = tf.reshape(pool3, (-1, 1 * 1 * 1024))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)


    ####### 辅助分类器 1 ########
    #### pool ####
    pool_1 = tf.nn.avg_pool(ince4a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #### conv ####
    conv_1 = tf.layers.conv2d(pool_1, filters=128, kernel_size=1, strides=1, padding='SAME')
    #### BN ####
    conv_1 = tf.layers.batch_normalization(conv_1)
    #### relu ####
    conv_1 = tf.nn.relu(conv_1)

    #### fc ####
    fc_1 = tf.layers.dense(conv_1, 1024)

    #### relu ####
    fc_1 = tf.nn.relu(fc_1)

    #### dropout ####
    drop_1 = tf.nn.dropout(fc_1, 0.3)

    #### out ####
    out_1 = tf.layers.dense(drop_1, n_classes)


    ####### 辅助分类器 2 ########
    #### pool ####
    pool_2 = tf.nn.avg_pool(ince4d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #### conv ####
    conv_2 = tf.layers.conv2d(pool_2, filters=128, kernel_size=1, strides=1, padding='SAME')
    #### BN ####
    conv_2 = tf.layers.batch_normalization(conv_2)
    #### relu ####
    conv_2 = tf.nn.relu(conv_2)

    #### fc ####
    fc_2 = tf.layers.dense(conv_2, 1024)

    #### relu ####
    fc_2 = tf.nn.relu(fc_2)

    #### dropout ####
    drop_2 = tf.nn.dropout(fc_2, 0.3)

    #### out ####
    out_2 = tf.layers.dense(drop_2, n_classes)

    return out, out_1, out_2


########## define model, loss and optimizer ##########
#### model pred 影像输出结果 ####
out, out_1, out_2 = Inception_v2(x, n_classes)

#### loss 损失计算 ####
## 阻断label的梯度流 ##
y = tf.stop_gradient(y)

## 主要输出结果 ##
cost_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))

## 辅助分类器1 ##
cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_1, labels=y))

## 辅助分类器2 ##
cost_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_2, labels=y))

## 定义损失 ##
cost = cost_real + 0.06 * cost_1 + 0.06 * cost_2

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(out), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

##################### train and evaluate model ##########################

########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(mnist.train.num_examples // batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples // batch_size_test):
        batch_x, batch_y = mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
