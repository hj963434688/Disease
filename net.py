import tensorflow.contrib.slim as slim
import tensorflow as tf
import nets.vgg as vgg
from nets import resnet_utils
from nets import resnet_v1
from nets import inception_v3
from nets import alexnet
from nets import inception_resnet_v2

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def le_net(inputs, num_class, is_training, regular):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 1e-4,
        'scale': True,
        'is_training': is_training
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.crelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):
        net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='conv1')
        net = slim.max_pool2d(net, 2, stride=2, scope='max-pool2')
        net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv3')
        net = slim.max_pool2d(net, 2, stride=2, scope='max-pool4')
        net = slim.flatten(net, scope='faltten')
        net = slim.fully_connected(net, 500, scope='fc5')

    net = slim.fully_connected(net, num_class, activation_fn=None, scope='output')
    return net


def model(inputs, tag, num_class, is_training, regular=0.0001):
    net = mean_image_subtraction(inputs)
    if tag == 0:
        return le_net(net, num_class, is_training, regular)
    if tag == 1:
        return vgg.vgg_16(net, num_classes=num_class, is_training=is_training)[0]
    if tag == 2:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, end_points = resnet_v1.resnet_v1_50(net, num_classes=num_class, is_training=is_training, scope='resnet_v1_50')
            return tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    if tag == 3:
        return inception_v3.inception_v3(net, num_classes=num_class, is_training=is_training, scope='inception_v3')[0]
    if tag == 4:
        return inception_resnet_v2.inception_resnet_v2(net, num_class, True, scope='incep_resnetv2')[0]