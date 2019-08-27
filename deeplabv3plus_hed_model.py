from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import preprocessing
import six
import pdb
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python import pywrap_tensorflow
from loss_al import attention_loss

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


def bias_variable(shape, initial):

    init = initial(shape)
    return tf.Variable(init)

def weight_variable(shape, initial):

    init = initial(shape)
    return tf.Variable(init)

def conv_layer(x, W_shape, b_shape=None, name=None,
               padding='SAME', use_bias=True, w_init=None, b_init=None):

  W = weight_variable(W_shape, w_init)
  tf.summary.histogram('weights_{}'.format(name), W)

  if use_bias:
    b = bias_variable([b_shape], b_init)
    tf.summary.histogram('biases_{}'.format(name), b)

  conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

  return conv + b if use_bias else conv

def deconv_layer(x, upscale, name, padding='SAME', w_init=None):
  x_shape = tf.shape(x)
  in_shape = x.shape.as_list()

  w_shape = [upscale * 2, upscale * 2, in_shape[-1], 1]
  strides = [1, upscale, upscale, 1]

  W = weight_variable(w_shape, w_init)
  tf.summary.histogram('weights_{}'.format(name), W)

  out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]]) * tf.constant(strides, tf.int32)
  deconv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)

  return deconv
  
  
def side_layer(inputs, name, upscale):
  with tf.variable_scope(name):
    in_shape = inputs.shape.as_list()
    w_shape = [1, 1, in_shape[-1], 1]
    classifier = conv_layer(inputs, w_shape, b_shape=1,
                                 w_init=tf.constant_initializer(),
                                 b_init=tf.constant_initializer(),
                                 name=name + '_reduction')
    classifier = deconv_layer(classifier, upscale=upscale,
                                   name='{}_deconv_{}'.format(name, upscale),
                                   w_init=tf.truncated_normal_initializer(stddev=0.1))
    return classifier


def hed_model_fn(inputs, is_training=False):
  inputs_size = tf.shape(inputs)[1:3]
  scope='vgg_16'
  with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      
      net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      side_1 = side_layer(net, "side_1", 1)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      side_2 = side_layer(net, "side_2", 2)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      side_3 = side_layer(net, "side_3", 4)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      side_4 = side_layer(net, "side_4", 8)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      side_5 = side_layer(net, "side_5", 16)
      
      #tf.image.crop_to_bounding_box
      side_1=tf.image.resize_bilinear(side_1, inputs_size, name='side_1')
      side_2=tf.image.resize_bilinear(side_2, inputs_size, name='side_2')
      side_3=tf.image.resize_bilinear(side_3, inputs_size, name='side_3')
      side_4=tf.image.resize_bilinear(side_4, inputs_size, name='side_4')
      side_5=tf.image.resize_bilinear(side_5, inputs_size, name='side_5')
      
      side_outputs = [side_1, side_2, side_3, side_4, side_5]

      w_shape = [1, 1, len(side_outputs), 1]
      fuse = conv_layer(tf.concat(side_outputs, axis=3),
                                    w_shape, use_bias=False,
                                    w_init=tf.constant_initializer(0.2))
      fuse= tf.image.resize_bilinear(fuse,inputs_size, name='side_fuse')
            
      outputs = side_outputs + [fuse]
      return outputs

        
def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]

        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        with tf.variable_scope("image_level_features"):
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net

def init_from_checkpoint(checkpoint_paths,variables_to_restore):
  var_to_shape_map={}
  for path in checkpoint_paths:
    # Read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    dict = reader.get_variable_to_shape_map()
    for key in dict:
      var_to_shape_map[key]=reader.get_tensor(key)
    #var_to_shape_map = {**var_to_shape_map, **dict}
  
  for key in variables_to_restore.keys():
    model_tsr=variables_to_restore[key]
    if key in var_to_shape_map.keys():
      local_tsr=var_to_shape_map[key]
      if model_tsr.get_shape().is_compatible_with(local_tsr.shape):
        model_tsr.assign(local_tsr)
      else:
        print("Shape of variable %s (%s) doesn't match with shape of "
            "tensor %s from checkpoint reader." % (
                key, str(model_tsr.get_shape()),
                str(local_tsr.get_shape())
            ))
    else:
      print(key + 'not exist in local checkpoint files')

      
def deeplab_v3_plus_generator(num_classes,
                              output_stride,
                              base_architecture,
                              pre_trained_model,
                              batch_norm_decay,
                              data_format='channels_last'):
  if data_format is None:
    pass

  if batch_norm_decay is None:
    batch_norm_decay = _BATCH_NORM_DECAY

  if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
    raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_50'.")

  if base_architecture == 'resnet_v2_50':
    base_model = resnet_v2.resnet_v2_50
  else:
    base_model = resnet_v2.resnet_v2_101

  def model(inputs, is_training):
    if data_format == 'channels_first':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      side_1, side_2, side_3, side_4, side_5, hed_fuse=hed_model_fn(inputs,is_training)
      inputs=tf.concat([hed_fuse,inputs],axis=3)
      logits, end_points = base_model(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride)

    if is_training:
      exclude = [base_architecture + '/logits', 'global_step','resnet_v2_101/conv1/weights','vgg_16/Variable','vgg_16/conv1/conv1_1/BatchNorm/beta','vgg_16/conv1/conv1_1/BatchNorm/gamma','vgg_16/conv1/conv1_1/BatchNorm/moving_mean','vgg_16/conv1/conv1_1/BatchNorm/moving_variance','vgg_16/conv1/conv1_2/BatchNorm/beta','vgg_16/conv1/conv1_2/BatchNorm/gamma','vgg_16/conv1/conv1_2/BatchNorm/moving_mean','vgg_16/conv1/conv1_2/BatchNorm/moving_variance','vgg_16/conv2/conv2_1/BatchNorm/beta','vgg_16/conv2/conv2_1/BatchNorm/gamma','vgg_16/conv2/conv2_1/BatchNorm/moving_mean','vgg_16/conv2/conv2_1/BatchNorm/moving_variance','vgg_16/conv2/conv2_2/BatchNorm/beta','vgg_16/conv2/conv2_2/BatchNorm/gamma','vgg_16/conv2/conv2_2/BatchNorm/moving_mean','vgg_16/conv2/conv2_2/BatchNorm/moving_variance','vgg_16/conv3/conv3_1/BatchNorm/beta','vgg_16/conv3/conv3_1/BatchNorm/gamma','vgg_16/conv3/conv3_1/BatchNorm/moving_mean','vgg_16/conv3/conv3_1/BatchNorm/moving_variance','vgg_16/conv3/conv3_2/BatchNorm/beta','vgg_16/conv3/conv3_2/BatchNorm/gamma','vgg_16/conv3/conv3_2/BatchNorm/moving_mean','vgg_16/conv3/conv3_2/BatchNorm/moving_variance','vgg_16/conv3/conv3_3/BatchNorm/beta','vgg_16/conv3/conv3_3/BatchNorm/gamma','vgg_16/conv3/conv3_3/BatchNorm/moving_mean','vgg_16/conv3/conv3_3/BatchNorm/moving_variance','vgg_16/conv4/conv4_1/BatchNorm/beta','vgg_16/conv4/conv4_1/BatchNorm/gamma','vgg_16/conv4/conv4_1/BatchNorm/moving_mean','vgg_16/conv4/conv4_1/BatchNorm/moving_variance','vgg_16/conv4/conv4_2/BatchNorm/beta','vgg_16/conv4/conv4_2/BatchNorm/gamma','vgg_16/conv4/conv4_2/BatchNorm/moving_mean','vgg_16/conv4/conv4_2/BatchNorm/moving_variance','vgg_16/conv4/conv4_3/BatchNorm/beta','vgg_16/conv4/conv4_3/BatchNorm/gamma','vgg_16/conv4/conv4_3/BatchNorm/moving_mean','vgg_16/conv4/conv4_3/BatchNorm/moving_variance',
      'vgg_16/conv5/conv5_1/BatchNorm/beta','vgg_16/conv5/conv5_1/BatchNorm/gamma','vgg_16/conv5/conv5_1/BatchNorm/moving_mean','vgg_16/conv5/conv5_1/BatchNorm/moving_variance','vgg_16/conv5/conv5_2/BatchNorm/beta','vgg_16/conv5/conv5_2/BatchNorm/gamma','vgg_16/conv5/conv5_2/BatchNorm/moving_mean','vgg_16/conv5/conv5_2/BatchNorm/moving_variance','vgg_16/conv5/conv5_3/BatchNorm/beta','vgg_16/conv5/conv5_3/BatchNorm/gamma','vgg_16/conv5/conv5_3/BatchNorm/moving_mean','vgg_16/conv5/conv5_3/BatchNorm/moving_variance',
      'vgg_16/side_1/Variable','vgg_16/side_2/Variable','vgg_16/side_3/Variable','vgg_16/side_4/Variable','vgg_16/side_5/Variable','vgg_16/side_fuse/Variable',]
      variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
      tf.train.init_from_checkpoint('init_weights/resnet_v2_101/resnet_v2_101.ckpt',
                                    {v.name.split(':')[0]: v for v in variables_to_restore if 'vgg' not in v.name})
                                    
      tf.train.init_from_checkpoint('init_weights/vgg_16.ckpt',
                                    {v.name.split(':')[0]: v for v in variables_to_restore if 'vgg' in v.name})

    inputs_size = tf.shape(inputs)[1:3]
    net = end_points[base_architecture + '/block4']
    encoder_output = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

    with tf.variable_scope("decoder"):
      with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
          with tf.variable_scope("low_level_features"):
            low_level_features = end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
            low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                   [1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]

          with tf.variable_scope("upsampling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

    return side_1, side_2, side_3, side_4, side_5, hed_fuse, logits

  return model


def deeplabv3_plus_model_fn(features, labels, mode, params):
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    labels = features['labels']
    edges = features['edges']
    features = features['images']
  print(mode)
  images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features),
      tf.uint8)

  network = deeplab_v3_plus_generator(params['num_classes'],
                                      params['output_stride'],
                                      params['base_architecture'],
                                      params['pre_trained_model'],
                                      params['batch_norm_decay'])
  side_1, side_2, side_3, side_4, side_5, hed_fuse, logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)
  pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

  pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                   [pred_classes, params['batch_size'], params['num_classes']],
                                   tf.uint8)

  side_output1 = tf.nn.sigmoid(side_1, name='side_output1')
  side_output2 = tf.nn.sigmoid(side_2, name='side_output2')
  side_output3 = tf.nn.sigmoid(side_3, name='side_output3')
  side_output4 = tf.nn.sigmoid(side_4, name='side_output4')
  side_output5 = tf.nn.sigmoid(side_5, name='side_output5')
  side_output_fuse = tf.nn.sigmoid(hed_fuse, name='side_output_fuse')
  pred_hed_side1 = tf.cast(tf.greater(side_output1, 0.8), tf.int32, name='pred_hed_side1')
  pred_hed_side2 = tf.cast(tf.greater(side_output2, 0.8), tf.int32, name='pred_hed_side2')
  pred_hed_side3 = tf.cast(tf.greater(side_output3, 0.8), tf.int32, name='pred_hed_side3')
  pred_hed_side4 = tf.cast(tf.greater(side_output4, 0.8), tf.int32, name='pred_hed_side4')
  pred_hed_side5 = tf.cast(tf.greater(side_output5, 0.8), tf.int32, name='pred_hed_side5')
  pred_hed_fuse = tf.cast(tf.greater(side_output_fuse, 0.8), tf.int32, name='fuse_predictions')
                                   
                                   
  predictions = {
      'classes': pred_classes,
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
      'decoded_labels': pred_decoded_labels,
      'logits':logits,
      'side_output1':side_1,
      'side_output2':side_2,
      'side_output3':side_3,
      'side_output4':side_4,
      'side_output5':side_5,
      'side_output_fuse':hed_fuse,
      
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    predictions_without_decoded_labels = predictions.copy()
    del predictions_without_decoded_labels['decoded_labels']

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions_without_decoded_labels)
        })
  gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                 [labels, params['batch_size'], params['num_classes']], tf.uint8)

  labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

  logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
  labels_flat = tf.reshape(labels, [-1, ])

  valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
     

  preds_flat = tf.reshape(pred_classes, [-1, ])
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

  predictions['valid_preds'] = valid_preds
  predictions['valid_labels'] = valid_labels
  predictions['confusion_matrix'] = confusion_matrix

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=valid_logits, labels=valid_labels)
  tf.identity(cross_entropy, name='cross_entropy')
  
  cost_side1,_ = attention_loss(side_1, edges, name='cross_entropy1')
  cost_side2,_ = attention_loss(side_2, edges, name='cross_entropy2')
  cost_side3,_ = attention_loss(side_3, edges, name='cross_entropy3')
  cost_side4,_ = attention_loss(side_4, edges, name='cross_entropy4')
  cost_side5,_ = attention_loss(side_5, edges, name='cross_entropy5')
  cost_side_fuse,_ = attention_loss(hed_fuse, edges, name='cross_entropy_fuse')

  
  accuracy_hed_side1 = tf.metrics.accuracy(
      edges, pred_hed_side1)
  tf.identity(accuracy_hed_side1[1], name='accuracy_hed_side1')
  tf.summary.scalar('accuracy_hed_side1', accuracy_hed_side1[1])

  accuracy_hed_side2 = tf.metrics.accuracy(
      edges, pred_hed_side2)
  tf.identity(accuracy_hed_side2[1], name='accuracy_hed_side2')
  tf.summary.scalar('accuracy_hed_side2', accuracy_hed_side2[1])

  accuracy_hed_side3 = tf.metrics.accuracy(
      edges, pred_hed_side3)
  tf.identity(accuracy_hed_side3[1], name='accuracy_hed_side3')
  tf.summary.scalar('accuracy_hed_side3', accuracy_hed_side3[1])

  accuracy_hed_side4 = tf.metrics.accuracy(
      edges, pred_hed_side4)
  tf.identity(accuracy_hed_side4[1], name='accuracy_hed_side4')
  tf.summary.scalar('accuracy_hed_side4', accuracy_hed_side4[1])

  accuracy_hed_side5 = tf.metrics.accuracy(
      edges, pred_hed_side5)
  tf.identity(accuracy_hed_side5[1], name='accuracy_hed_side5')
  tf.summary.scalar('accuracy_hed_side5', accuracy_hed_side5[1])
  
  
  accuracy_hed_fuse = tf.metrics.accuracy(
      edges, pred_hed_fuse)
  tf.identity(accuracy_hed_fuse[1], name='accuracy_hed_fuse')
  tf.summary.scalar('accuracy_hed_fuse', accuracy_hed_fuse[1])
  
  
  if not params['freeze_batch_norm']:
    train_var_list = [v for v in tf.trainable_variables()]
  else:
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]

  with tf.variable_scope("total_loss"):
    loss = cross_entropy + cost_side1 + cost_side2 + cost_side3 + cost_side4 + cost_side5 + cost_side_fuse + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf.summary.image('images',
                     tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
                     max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

    global_step = tf.train.get_or_create_global_step()

    if params['learning_rate_policy'] == 'piecewise':
      initial_learning_rate = 0.1 * params['batch_size'] / 128
      batches_per_epoch = params['num_train'] / params['batch_size']
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
      values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
          tf.cast(global_step, tf.int32), boundaries, values)
    elif params['learning_rate_policy'] == 'poly':
      learning_rate = tf.train.polynomial_decay(
          params['initial_learning_rate'],
          tf.cast(global_step, tf.int32) - params['initial_global_step'],
          params['max_iter'], params['end_learning_rate'], power=params['power'])
    else:
      raise ValueError('Learning rate policy must be "piecewise" or "poly"')

    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
  metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou, 'accuracy_hed_side1':accuracy_hed_side1,
  'accuracy_hed_side2':accuracy_hed_side2,'accuracy_hed_side3':accuracy_hed_side3,
  'accuracy_hed_side4':accuracy_hed_side4,'accuracy_hed_side5':accuracy_hed_side5,
  'accuracy_hed_fuse':accuracy_hed_fuse}

  tf.identity(accuracy[1], name='train_px_accuracy')
  tf.summary.scalar('train_px_accuracy', accuracy[1])

  def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

  train_mean_iou = compute_mean_iou(mean_iou[1])

  tf.identity(train_mean_iou, name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', train_mean_iou)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
