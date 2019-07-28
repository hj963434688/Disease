import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import data_genertor
import numpy as np
import time
import net

batch_size = 64
crop_size = 224
number_readers = 8

lr = 0.01
lr_decay = 0.94
decay_steps = 2000
max_step = 80000

moving_decay = 0.997
regularization_rate = None

pre_train = False
pre_train_path = None

save_step = 2000
val_step = 500
log_step = 10

train_paths = ['./expe_model/v_lenet_apple/', './expe_model/v_vgg_all/', './expe_model/v_resnet_all/', './expe_model/v_google_all/',
               './expe_model/v_incep_resnet_all/']
tag = 4
num_class = 61
checkpoint_path = train_paths[tag]
model_name = 'model.ckpt'
whether_resort = True


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not tf.gfile.Exists(checkpoint_path):
        tf.gfile.MakeDirs(checkpoint_path)
    else:
        if not whether_resort:
            tf.gfile.DeleteRecursively(checkpoint_path)
            tf.gfile.MakeDirs(checkpoint_path)

    x = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3], name='input_images')
    y_ = tf.placeholder(tf.float32, shape=[None, num_class], name='input_labels')

    y = net.model(x, is_training=True, tag=tag, num_class=num_class, regular=regularization_rate)

    print(y.shape)
    print(y_.shape)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    acc_sum_train = tf.summary.scalar('accuracy_train', accuracy)
    acc_sum_val = tf.summary.scalar('accuracy_val', accuracy)
    cross_sum = tf.summary.scalar('cross_entropy', cross_entropy)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=decay_steps, decay_rate=lr_decay, staircase=True)
    lr_sum = tf.summary.scalar('learning_rate', learning_rate)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step)

    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([train_step, update_ops]):
        train_op = tf.no_op(name='train')

    #
    summary_train_op = tf.summary.merge([acc_sum_train, cross_sum, lr_sum])
    summary_val_op = tf.summary.merge([acc_sum_val])
    summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

    # saver = tf.train.Saver(tf.global_variables())
    saver = tf.train.Saver(max_to_keep=1)
    init = tf.global_variables_initializer()

    if pre_train_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(pre_train_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU setting
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if whether_resort:
            print('continue training from previous checkpoint', checkpoint_path)
            ckpt = tf.train.latest_checkpoint(checkpoint_path)
            try:
                current_step = int(ckpt.split('-')[1]) + 1
            except:
                current_step = 0
            print(current_step)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            current_step = 0
            if pre_train_path is not None:
                variable_restore_op(sess)
        data_gene = data_genertor.get_batch(num_workers=number_readers, crop_size=crop_size, batch_size=batch_size, class_num=num_class)

        start = time.time()
        train_time = []
        for step in range(current_step, max_step + 1):

            data = next(data_gene)

            train_star = time.time()
            _, loss, acc = sess.run([train_op, cross_entropy, accuracy], feed_dict={x: data[0], y_: data[1]})
            train_time.append(time.time() - train_star)

            if np.isnan(loss):
                print('loss diverged, stop training')
                break

            if step % log_step == 0:
                avg_time_per_step = (time.time() - start) / log_step
                avg_traintime_per_step = sum(train_time) / log_step
                avg_examples_per_second = (log_step * batch_size) / (time.time() - start)
                train_time = []
                start = time.time()
                print(
                    'step {:06d}, loss {:.4f}, acc {:.4f}, all:{:.2f}sec/step, train:{:.2f}sec/ste, {:.2f} examples/second'.format(
                        step, loss, acc, avg_time_per_step, avg_traintime_per_step, avg_examples_per_second))
                summary_str = sess.run(summary_train_op, feed_dict={x: data[0], y_: data[1]})
                summary_writer.add_summary(summary_str, global_step=step)

            if step % save_step == 0:
                saver.save(sess, checkpoint_path + 'model.ckpt', global_step=step)

            if step % val_step == 0:
                val_data = data_genertor.get_val_batch(crop_size=crop_size, batch_size=batch_size, class_num=num_class)
                summary_str, val_acc = sess.run([summary_val_op, accuracy], feed_dict={x: val_data[0], y_: val_data[1]})
                print('after {} step, val_acc is {}'.format(step, val_acc))
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    main()