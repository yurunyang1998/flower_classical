import tensorflow as tf
from build_model.py import *
import  readTfrecordfile


img, lab = readTfrecordfile.read_and_decode(r'D:\python_practice\跟着YouTube学tf\.i'
                                            r'dea\flowers2_tfrecord.tfrecord')




with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum( y * tf.log(h_func2),
                                                  reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(h_func2,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#img,lab = read_and_decode(r'D:\python_practice\跟着YouTube学tf\.idea\flowers2_tfrecord.tfrecord')
img_batch , lab_batch = tf.train.batch([img,lab],batch_size=5)

init = tf.initialize_all_variables()

with tf.Session() as sess :
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    try :
        while not coord.should_stop():
            sess.run([img_batch, lab_batch])
            sess.run(train_step, feed_dict={x: img_batch, y: lab_batch, keep_drop: 0.75})
            print(123)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()