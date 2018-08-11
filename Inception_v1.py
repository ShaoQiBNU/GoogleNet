########## load packages ##########
import tensorflow as tf

##################### load data ##########################

'''
导入自己的数据
'''

########## set net hyperparameters ##########
learning_rate=0.0001

epochs=20
batch_size_train=128
batch_size_test=100

display_step=20

########## set net parameters ##########
#### img shape:224*224 ####

#### classes ####
n_classes=1000

#### dropout probability
dropout=0.6


# Handle Dimension Ordering for different backends
'''
img_input_shape=(224, 224, 3)
concat_axis = 3

img_input_shape=(3, 224, 224)
concat_axis=1
'''
global concat_axis

concat_axis=3

########## placeholder ##########
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])


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
    f1, f3_r, f3, f5_r, f5, proj=filters

    ######## conv # 1x1 ########
    conv1 = tf.layers.conv2d(input,filters=f1,kernel_size=1,strides=1,padding='SAME', activation=tf.nn.relu)


    ######## conv # 3x3 ########
    #### conv # 3x3 reduce ####
    conv3r = tf.layers.conv2d(input,filters=f3_r,kernel_size=1,strides=1,padding='SAME', activation=tf.nn.relu)

    #### conv # 3x3 reduce ####
    conv3 = tf.layers.conv2d(conv3r,filters=f3,kernel_size=3,strides=1,padding='SAME', activation=tf.nn.relu)


    ######## conv # 5x5 ########
    #### conv # 5x5 reduce ####  
    conv5r = tf.layers.conv2d(input,filters=f5_r,kernel_size=1,strides=1,padding='SAME', activation=tf.nn.relu)

    #### conv # 5x5 ####
    conv5 = tf.layers.conv2d(conv5r,filters=f5,kernel_size=3,strides=1,padding='SAME', activation=tf.nn.relu)


    ######## pool proj ########
    #### pool ####
    pool1 = tf.nn.max_pool(input,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')

    #### proj ####
    convproj = tf.layers.conv2d(pool1,filters=proj,kernel_size=1,strides=1,padding='SAME', activation=tf.nn.relu)


    ######## 连接 ########
    out = tf.concat([conv1, conv3, conv5, convproj],concat_axis)

    return out

######### GoogleNet Inception v1 ##########
def Inception_v1(x,n_classes):

    '''
    x: input image
    n_classes: 类别，可根据数据集调整

    return:
           out:   主要输出
           out_1: 辅助分类器1输出
           out_2: 辅助分类器2输出

    '''

    ####### reshape input picture ########
    x=tf.reshape(x,shape=[-1,224,224,1])


    ####### first conv ########
    #### conv 1 ####
    conv1=tf.layers.conv2d(x,filters=64,kernel_size=7,strides=2,padding='SAME', activation=tf.nn.relu)


    ####### max pool ########
    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### second conv ########
    #### conv 2_1 ####
    conv2=tf.layers.conv2d(pool1,filters=192,kernel_size=1,strides=1,padding='VALID', activation=tf.nn.relu)
    
    #### conv 2_2 ####
    conv2=tf.layers.conv2d(conv2,filters=192,kernel_size=3,strides=1,padding='SAME', activation=tf.nn.relu)


    ####### max pool ########
    pool2=tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### inception 3a ########
    ince3a=inception(pool2, [64, 96, 128, 16, 32, 32])

    ####### inception 3b ########
    ince3b=inception(ince3a, [128, 128, 192, 32, 96, 64])


    ####### max pool ########
    pool3=tf.nn.max_pool(ince3b,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### inception 4a ########
    ince4a=inception(pool3, [192, 96, 208, 16, 48, 64])

    ####### inception 4b ########
    ince4b=inception(ince4a, [160, 112, 224, 24, 64, 64])

    ####### inception 4c ########
    ince4c=inception(ince4b, [128, 128, 256, 24, 64, 64])

    ####### inception 4d ########
    ince4d=inception(ince4c, [112, 144, 288, 32, 64, 64])

    ####### inception 4e ########
    ince4e=inception(ince4d, [256, 160, 320, 32, 128, 128])


    ####### max pool ########
    pool4=tf.nn.max_pool(ince4e,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### inception 5a ########
    ince5a=inception(pool4, [256, 160, 320, 32, 128, 128])

    ####### inception 5b ########
    ince5b=inception(ince5a, [384, 192, 384, 48, 128, 128])


    ####### average pool ########
    pool5=tf.nn.avg_pool(ince5b,ksize=[1,7,7,1],strides=[1,1,1,1],padding='VALID')


    ## dropout ##
    drop1=tf.nn.dropout(pool5, dropout)

    ###### flatten 影像展平 ########
    flatten = tf.reshape(drop1, (-1, 1*1*1024))

    ####### out 输出，1000类 可根据数据集进行调整 ########
    out=tf.layers.dense(flatten,n_classes)

    ####### softmax ########
    out=tf.nn.softmax(out)


    ####### 辅助分类器 1 ########
    #### pool ####
    pool_1=tf.nn.avg_pool(ince4a,ksize=[1,5,5,1],strides=[1,3,3,1],padding='VALID')

    #### conv ####
    conv_1=tf.layers.conv2d(pool_1,filters=128,kernel_size=1,strides=1,padding='SAME', activation=tf.nn.relu)

    #### fc ####
    fc_1=tf.layers.dense(conv_1,1024)

    #### relu ####
    fc_1=tf.nn.relu(fc_1)

    #### dropout ####
    drop_1=tf.nn.dropout(fc_1, 0.3)

    #### out ####
    out_1=tf.layers.dense(drop_1,n_classes)

    #### softmax ####
    out_1=tf.nn.softmax(out_1)


    ####### 辅助分类器 2 ########
    #### pool ####
    pool_2=tf.nn.avg_pool(ince4d,ksize=[1,5,5,1],strides=[1,3,3,1],padding='VALID')

    #### conv ####
    conv_2=tf.layers.conv2d(pool_2,filters=128,kernel_size=1,strides=1,padding='SAME', activation=tf.nn.relu)

    #### fc ####
    fc_2=tf.layers.dense(conv_2,1024)

    #### relu ####
    fc_2=tf.nn.relu(fc_2)

    #### dropout ####
    drop_2=tf.nn.dropout(fc_2, 0.3)

    #### out ####
    out_2=tf.layers.dense(drop_2,n_classes)

    #### softmax ####
    out_2=tf.nn.softmax(out_2)


    return out, out_1, out_2


########## define model, loss and optimizer ##########
#### model pred 影像输出结果 ####
out, out_1, out_2=Inception_v1(x,n_classes)


#### loss 损失计算 ####
## 主要输出结果 ##
cost_real=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

## 辅助分类器1 ##
cost_1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_1, labels=y))

## 辅助分类器2 ##
cost_2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_2, labels=y))

## 损失 ##
cost=cost_real+0.3*cost_1+0.3*cost_2


#### optimization 优化 ####
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred=tf.equal(tf.argmax(out,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    #### epoch 世代循环 ####
    for epoch in range(epochs+1):

        #### iteration ####
        for _ in range(iterations):

            step += 1

            ##### get x,y #####
            batch_x, batch_y=# 导入自己的数据 batch

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_x, y: batch_y})
                print("Epoch "+ str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))


    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(iterations):
        batch_x,batch_y=mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))