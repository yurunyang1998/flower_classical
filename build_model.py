import  tensorflow as tf
from  define_model_function import  *
###change
x = tf.placeholder(tf.float32, shape=[None,None,2352])
y = tf.placeholder(tf.float32, shape=[None, 5])
x_image = tf.reshape(x,shape=[-1,28,28,3])


W_conv1 = weight_varibles([5,5,3,32])
biases_conv1 = bias_varibles([32])
conv_layer1 = tf.nn.relu(conv_layer(x_image, W_conv1)+biases_conv1)
pool_layer1 = max_pool(conv_layer1)


W_conv2 = weight_varibles([5,5,32,64])
biases_conv2 = bias_varibles([64])
conv_layer2 = tf.nn.relu(conv_layer(pool_layer1, W_conv2)+biases_conv2)
pool_layer2 = max_pool(conv_layer2)


W_func1 = weight_varibles([7*7*64, 512])
b_fc1 = tf.Variable([512],dtype=tf.float32)
h_pool2_flat = tf.reshape(pool_layer2,[-1,7*7*64])
#h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_func1)+b_fc1)
h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_func1)+b_fc1)


keep_drop = tf.placeholder(tf.float32)
h_fun1_drop = tf.nn.dropout(h_func1, keep_drop)


W_func2 = weight_varibles([512, 5])
b_fc2 = bias_varibles([5])
h_func2 = tf.nn.softmax(tf.matmul(h_fun1_drop, W_func2)+b_fc2)

