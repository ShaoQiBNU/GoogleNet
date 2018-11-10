GoogleNet网络说明
================

# 一. 概述

> Google Inception Net首次出现在ILSVRC 2014的比赛中，以较大优势取得了第一名。那届比赛中的Inception Net通常被称为Inception V1，它最大的特点是控制了计算量和参数量的同时，获得了非常好的分类性能——top-5错误率6.67%，只有AlexNet的一半不到。Inception V1有22层深，比AlexNet的8层或者VGGNet的19层还要更深。但其计算量只有15亿次浮点运算，同时只有500万的参数量，仅为AlexNet参数量（6000万）的1/12，却可以达到远胜于AlexNet的准确率，可以说是非常优秀并且非常实用的模型。Inception V1降低参数量的目的有两点，第一，参数越多模型越庞大，需要供模型学习的数据量就越大，而目前高质量的数据非常昂贵；第二，参数越多，耗费的计算资源也会更大。

> Inception V1参数少但效果好的原因除了模型层数更深、表达能力更强外，还有两点：一是去除了最后的全连接层，用全局平均池化层（即将图片尺寸变为1 x 1）来取代它。全连接层几乎占据了AlexNet或VGGNet中90%的参数量，而且会引起过拟合，去除全连接层后模型训练更快并且减轻了过拟合。用全局平均池化层取代全连接层的做法借鉴了NetworkIn Network（以下简称NIN）论文。二是Inception V1中精心设计的InceptionModule提高了参数的利用效率，其结构如图1所示。这一部分也借鉴了NIN的思想，形象的解释就是Inception Module本身如同大网络中的一个小网络，其结构可以反复堆叠在一起形成大网络。不过Inception V1比NIN更进一步的是增加了分支网络，NIN则主要是级联的卷积层和MLPConv层。一般来说卷积层要提升表达能力，主要依靠增加输出通道数，但副作用是计算量增大和过拟合。每一个输出通道对应一个滤波器，同一个滤波器共享参数，只能提取一类特征，因此一个输出通道只能做一种特征处理。而NIN中的MLPConv则拥有更强大的能力，允许在输出通道之间组合信息，因此效果明显。可以说，MLPConv基本等效于普通卷积层后再连接1 x 1的卷积和ReLU激活函数。

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/1.png)

# 二. InceptionModule

> Inception Module的基本结构如图1所示，常采用右边model，为了降维，有4个分支：第一个分支对输入进行1 x 1的卷积，这其实也是NIN中提出的一个重要结构。1 x 1的卷积是一个非常优秀的结构，它可以跨通道组织信息，提高网络的表达能力，同时可以对输出通道升维和降维。可以看到Inception Module的4个分支都用到了1 x 1卷积，来进行低成本（计算量比3 x 3小很多）的跨通道的特征变换。  第二个分支先使用了1 x 1卷积，然后连接3 x 3卷积，相当于进行了两次特征变换。  第三个分支类似，先是1 x 1的卷积，然后连接5 x 5卷积。  最后一个分支则是3 x 3最大池化后直接使用1 x 1卷积。有的分支只使用1 x 1卷积，有的分支使用了其他尺寸的卷积时也会再使用1 x 1卷积，这是因为1 x 1卷积的性价比很高，用很小的计算量就能增加一层特征变换和非线性化。Inception Module的4个分支在最后通过一个聚合操作合并（在输出通道数这个维度上聚合）。

> Inception Module中包含了3种不同尺寸的卷积和1个最大池化，增加了网络对不同尺度的适应性，这一部分和Multi-Scale的思想类似。早期计算机视觉的研究中，受灵长类神经视觉系统的启发，Serre使用不同尺寸的Gabor滤波器处理不同尺寸的图片，Inception V1借鉴了这种思想。Inception V1的论文中指出，InceptionModule可以让网络的深度和宽度高效率地扩充，提升准确率且不致于过拟合。

# 三. Inception Net

## (一) Inception V1

> Inception Net V1结构如图所示，

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/4.png)

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/2.png)

> Inception Net V1 有22层深，每一个卷积层都有relu激活函数，除了最后一层的输出，其中间节点的分类效果也很好。因此在Inception Net V1 中，还使用到了辅助分类节点（auxiliary classifiers），即将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终loss中。这样相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个Inception Net的训练很有裨益。

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/3.png)


> Inception V1也使用了Multi-Scale、Multi-Crop等数据增强方法，并在不同的采样数据上训练了7个模型进行融合，得到了最后的ILSVRC 2014的比赛成绩——top-5错误率6.67%。


### 代码

> 应用Inception V1结构实现MNIST判别，由于MNIST的图像大小为28x28，到第四次pool时，影像已经成为1 x 1 x 832，所以两个辅助分类器的pool的ksize和strides做了调整，代码如下：

```python
########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_sets", one_hot=True)

########## set net hyperparameters ##########
learning_rate = 0.0001

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
def inception(input, filters):
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
    conv1 = tf.layers.conv2d(input, filters=f1, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)

    ######## conv # 3x3 ########
    #### conv # 3x3 reduce ####
    conv3r = tf.layers.conv2d(input, filters=f3_r, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)

    #### conv # 3x3 reduce ####
    conv3 = tf.layers.conv2d(conv3r, filters=f3, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)

    ######## conv # 5x5 ########
    #### conv # 5x5 reduce ####
    conv5r = tf.layers.conv2d(input, filters=f5_r, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)

    #### conv # 5x5 ####
    conv5 = tf.layers.conv2d(conv5r, filters=f5, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)

    ######## pool proj ########
    #### pool ####
    pool1 = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    #### proj ####
    convproj = tf.layers.conv2d(pool1, filters=proj, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)

    ######## 连接 ########
    out = tf.concat([conv1, conv3, conv5, convproj], concat_axis)

    return out


######### GoogleNet Inception v1 ##########
def Inception_v1(x, n_classes):
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
    conv1 = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='SAME', activation=tf.nn.relu)

    ####### max pool ########
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### second conv ########
    #### conv 2_1 ####
    conv2 = tf.layers.conv2d(pool1, filters=192, kernel_size=1, strides=1, padding='VALID', activation=tf.nn.relu)

    #### conv 2_2 ####
    conv2 = tf.layers.conv2d(conv2, filters=192, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)

    ####### max pool ########
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### inception 3a ########
    ince3a = inception(pool2, [64, 96, 128, 16, 32, 32])

    ####### inception 3b ########
    ince3b = inception(ince3a, [128, 128, 192, 32, 96, 64])

    ####### max pool ########
    pool3 = tf.nn.max_pool(ince3b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### inception 4a ########
    ince4a = inception(pool3, [192, 96, 208, 16, 48, 64])

    ####### inception 4b ########
    ince4b = inception(ince4a, [160, 112, 224, 24, 64, 64])

    ####### inception 4c ########
    ince4c = inception(ince4b, [128, 128, 256, 24, 64, 64])

    ####### inception 4d ########
    ince4d = inception(ince4c, [112, 144, 288, 32, 64, 64])

    ####### inception 4e ########
    ince4e = inception(ince4d, [256, 160, 320, 32, 128, 128])

    ####### max pool ########
    pool4 = tf.nn.max_pool(ince4e, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### inception 5a ########
    ince5a = inception(pool4, [256, 160, 320, 32, 128, 128])

    ####### inception 5b ########
    ince5b = inception(ince5a, [384, 192, 384, 48, 128, 128])

    ####### average pool ########
    pool5 = tf.nn.avg_pool(ince5b, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    ## dropout ##
    drop1 = tf.nn.dropout(pool5, dropout)

    ###### flatten 影像展平 ########
    flatten = tf.reshape(drop1, (-1, 1 * 1 * 1024))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)


    ####### 辅助分类器 1 ########
    #### pool ####
    pool_1 = tf.nn.avg_pool(ince4a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #### conv ####
    conv_1 = tf.layers.conv2d(pool_1, filters=128, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)

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
    conv_2 = tf.layers.conv2d(pool_2, filters=128, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)

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
out, out_1, out_2 = Inception_v1(x, n_classes)

#### loss 损失计算 ####
## 主要输出结果 ##
cost_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

## 辅助分类器1 ##
cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_1, labels=y))

## 辅助分类器2 ##
cost_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_2, labels=y))

## 损失 ##
cost = cost_real + 0.3 * cost_1 + 0.3 * cost_2

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
```

## (二) Inception V2

> Inception v2的网络在Inception v1的基础上，进行了改进，一方面了加入了BN层（放在激活函数之前，卷积之后），减少了Internal Covariate Shift（内部神经元分布的改变），使每一层的输出都规范化到一个N(0, 1)的高斯，还去除了Dropout、LRN等结构；另外一方面学习VGG用2个3x3的卷积替代inception模块中的5x5卷积，既降低了参数数量，又加速计算，如图所示：

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/6.png)

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/7.png)

> Inception v2的结构如图所示：

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/5.png)

> 其中，pass through代表不做proj，直接将pool之后的结果输出；由于Inception v2移除了Inception v1在inception层之间的全局pool层，所以当数据由inception 3c ---> inception 4a时，四个分支的stride变为2，inception 4e ---> inception 5a时，四个分支的stride变为2。Inception v2的结构与Inception v1的其他区别见论文。

### 代码

> 应用Inception V2结构实现MNIST判别，由于MNIST的图像大小为28x28，到inception 4时，影像已经成为2 x 2 x 576，所以两个辅助分类器的pool的ksize和strides做了调整，代码如下：

```python
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


```

## (三) Inception V3

### 代码




## (四) Inception V4
> inception v4的网络结构设计如下：

![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/9.png)

### 各个子模块的结构如下：
#### Stem
![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/10.png)

#### Inception A
![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/11.png)

#### Reduction A
![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/12.png)

#### Inception B
![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/13.png)

#### Reduction B
![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/14.png)

#### Inception C
![image](https://github.com/ShaoQiBNU/GoogleNet/blob/master/images/15.png)

### 代码


# 四. GoogleNet总结

> Inception Net各类型网络总结如下：

> 1. Inception v1的网络，打破了常规的卷积层串联的模式，将1x1，3x3，5x5的卷积层和3x3的pooling池化层并联组合后concatenate组装在一起的设计思路；

> 2. Inception v2的网络在Inception v1的基础上，进行了改进，一方面了加入了BN层，减少了Internal Covariate Shift（内部神经元分布的改变），使每一层的输出都规范化到一个N(0, 1)的高斯，还去除了Dropout、LRN等结构；另外一方面学习VGG用2个3x3的卷积替代inception模块中的5x5卷积，既降低了参数数量，又加速计算；

> 3. Inception v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1）。这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，可以处理更多更丰富的空间特征，增加特征多样性。还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块；

> 4. Inception v4结合了微软的ResNet，发现ResNet的结构可以极大地加速训练，同时性能也有提升，得到一个Inception-ResNet v2网络，同时还设计了一个更深更优化的Inception v4模型，能达到与Inception-ResNet v2相媲美的性能。

