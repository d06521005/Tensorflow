# 書本範例： TensorFlow + Keras 深度學習人工智慧實務應用

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline




# Read data from tensorflow
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Function to create parameter
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name='b'))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# ksize = [1, height, width, 1]
# strides = [1, stride, stride, 1]
# padding = 'SAME' , padding '0' for keeping same size 



# <<<<<< Building Model >>>>>>>>>>
# using 'tf.name_scope' function to build computational graph
# Input Layer
with tf.name_scope('Input_Layer'):
    x = tf.placeholder('float', shape=[None, 784], name='x')
    x_image = tf.reshape(x, [-1,28,28,1]) # if color image [-1,m,n,3]
    

# Convolution Layer 1
with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,1,16])
    # [filter_h, filter_w, z_dim, number of filter]
    b1 = bias([16])
    Conv1 = conv2d(x_image, W1) + b1 # x_image as input
    C1_Conv = tf.nn.relu(Conv1 )
    # output shape 28x28x16
    
# Max pool Layer 1
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)
    # output shape 14x14x16

    
# Convolution Layer 2    
with tf.name_scope('C2_Conv'):
    W2 = weight([5,5,16,36])
    # [filter_h, filter_w, z_dim, number of filter]
    b2 = bias([36])
    Conv2=conv2d(C1_Pool, W2)+ b2
    C2_Conv = tf.nn.relu(Conv2)
    # output shape 14x14x36
    
    
# Max pool Layer 2
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv) 
    # output shape 7x7x36
    
# Fully Connected Layer    
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])
    
    
# Hidden Layer
with tf.name_scope('D_Hidden_Layer'):
    W3= weight([1764, 128])
    b3= bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3)+b3)
    D_Hidden_Dropout= tf.nn.dropout(D_Hidden, keep_prob=0.8)
    

# Output Layers
with tf.name_scope('Output_Layer'):
    W4 = weight([128,10])
    b4 = bias([10])
    y_predict= tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4)+b4)
    

# Set the optimizer method for training model
with tf.name_scope("optimizer"):
    y_label = tf.placeholder("float", shape=[None, 10],  name="y_label")
    
    loss_function = tf.reduce_mean(
                         tf.nn.softmax_cross_entropy_with_logits
                                    (logits=y_predict, labels=y_label)
                                   )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)
    
    
# Accuracy
with tf.name_scope("evaluate_model"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    

#<<<<<<<<  Training Model  >>>>>>>>
# Set training parameters
trainEpochs = 5
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize) # iter of one epoch
epoch_list=[]
accuracy_list=[]
loss_list=[];

# Record time
from time import time
startTime=time()


# Using Session() to execute computational graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Traing Model 
for epoch in range(trainEpochs):
    # Period of epoch
    
    for i in range(totalBatchs):
        # iter times of each epoch
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={x: batch_x, y_label: batch_y})
        # directly run optimizer to training model (run the computational graph)
        
    
    loss, acc = sess.run([loss_function, accuracy],
                          feed_dict={x: mnist.validation.images, 
                                     y_label: mnist.validation.labels}
                        )

    # Record 
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)    
    
    # Print information
    print("Train Epoch:", '%02d' %(epoch+1),
          "Loss=","{:.9f}".format(loss),
          " Accuracy=",acc)
    
duration = time() - startTime
print("Train Finished takes:",duration)


# <<<<< ploting >>>>>>>>>
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list, loss_list, label = 'loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')

plt.plot(epoch_list, accuracy_list,label="accuracy" )
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


# <<<<<< Accuracy >>>>>>>>
print("Accuracy:", 
      sess.run(accuracy,feed_dict={x: mnist.test.images,
                                   y_label: mnist.test.labels}))
                                   
prediction_result=sess.run(tf.argmax(y_predict,1),
                           feed_dict={x: mnist.test.images ,
                                      y_label: mnist.test.labels})
                                      

# <<<<<<  show images >>>>>>>>                                      
def show_images_labels_predict(images,labels,prediction_result):
    fig = plt.gcf()
    fig.set_size_inches(8, 10)
    for i in range(0, 10):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(np.reshape(images[i],(28, 28)), 
                  cmap='binary')
        ax.set_title("label=" +str(np.argmax(labels[i]))+
                     ",predict="+str(prediction_result[i])
                     ,fontsize=9) 
    plt.show()
    
# Run 'show_images_labels_predict'    
show_images_labels_predict(mnist.test.images,mnist.test.labels,prediction_result)


# <<<<< Show the mistake image>>>>>>>>>>>
def show_images_labels_predict_error(images,labels,prediction_result):
    fig = plt.gcf()
    fig.set_size_inches(8, 10)
    i=0;j=0
    while i<10:
        if prediction_result[j]!=np.argmax(labels[j]):
            ax=plt.subplot(5,5, 1+i)
            ax.imshow(np.reshape(images[j],(28, 28)), 
                      cmap='binary')
            ax.set_title("j="+str(j)+
                         ",Ans:" +str(np.argmax(labels[j]))+
                         ",pred:"+str(prediction_result[j])
                         ,fontsize=9) 
            i=i+1  
        j=j+1
    plt.show()
    

for i in range(500):
    if prediction_result[i]!=np.argmax(mnist.test.labels[i]):
        print("i="+str(i)+
              "   label=",np.argmax(mnist.test.labels[i]),
              "predict=",prediction_result[i])

# Run 'show_images_labels_predict_error'
show_images_labels_predict_error(mnist.test.images,mnist.test.labels,prediction_result)


# <<<<<<  Save Model  >>>>>>
saver = tf.train.Saver()
save_path = saver.save(sess, "saveModel/CNN_model1")
print("Model saved in file: %s" % save_path)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/CNN',sess.graph)

#sess.close()
