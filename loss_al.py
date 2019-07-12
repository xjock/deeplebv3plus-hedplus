import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops

def attention_loss(logits, label, beta=4, gamma=0.5, name='attention_loss'):
    """
    Implements Attention Loss in DOOBNet: Deep Object Occlusion Boundary Detection from an Image
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    alpha = count_neg / (count_neg + count_pos)
    
    # Equation [2] divide by 1 - alpha
    pos_weight = alpha / (1 - alpha)
    
    '''
    refer to weighted_cross_entropy_with_logits_v2
      labels * -log(sigmoid(logits)) * pos_weight +
          (1 - labels) * -log(1 - sigmoid(logits))
    For brevity, let `x = logits`, `z = labels`, `q = pos_weight`.
    The loss is:
        qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      
    to ensure stability and avoid overflow, the implementation uses
      log(1 + exp(-x)) == log(1 + exp(-abs(x))) + max(-x, 0)

    '''
    
    beta = tf.cast(beta, tf.float32)
    
    sigma = math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) + nn_ops.relu(-logits)
    
    p = tf.nn.sigmoid(logits)
    
    cost = pos_weight * y * sigma * tf.pow(beta,tf.pow(1 - p, gamma)) + (1 - y) * (logits + sigma) * tf.pow(beta,tf.pow(p, gamma))

    '''
    #####weighted_cross_entropy_with_logits_v2###################################
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(y, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    log_weight = 1 + (pos_weight - 1) * labels
    cost = math_ops.add(
        (1 - labels) * logits,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) + nn_ops.relu(-logits)),
        name=name)
    #############################################################################
    '''
    
    # Multiply by 1 - alpha
    cost = tf.reduce_mean(cost * (1 - alpha))

    # check if image has no edge pixels return 0 else return complete error function
    return cost, tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
