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

    alpha = count_neg / (count_neg + count_pos)
    
    pos_weight = alpha / (1 - alpha)
       
    beta = tf.cast(beta, tf.float32)
    
    sigma = math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) + nn_ops.relu(-logits)
    
    p = tf.nn.sigmoid(logits)
    
    eps = 1e-14
    
    p_clip = tf.clip_by_value(p, eps, 1.0-eps)

    cost = pos_weight * y * sigma * tf.pow(beta,tf.pow(1 - p_clip, gamma)) + (1 - y) * (logits + sigma) * tf.pow(beta,tf.pow(p_clip, gamma))
   
    cost = tf.reduce_mean(cost)

    return cost, tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
