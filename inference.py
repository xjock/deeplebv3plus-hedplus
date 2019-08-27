from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import deeplabv3plus_hed_model
import preprocessing
import dataset_util
from PIL import Image
import matplotlib.pyplot as plt
import pdb
from tensorflow.python import debug as tf_debug
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='v/test/top',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./v_infer.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 5


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplabv3plus_hed_model.deeplabv3_plus_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })

  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]

  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=pred_hooks)

  output_dir = FLAGS.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  for pred_dict, image_path in zip(predictions, image_files):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = image_basename + '.jpg'
    path_to_output = os.path.join(output_dir, output_filename)

    print("generating:", path_to_output)
    mask = pred_dict['decoded_labels']
    raw_mask=pred_dict['classes']
    cv2.imwrite(os.path.join(output_dir,image_basename+'.png'),raw_mask)
   
    side1_mask=pred_dict['side_output1']
    cv2.imwrite(os.path.join(output_dir,image_basename+'_side1.png'),side1_mask)
    side2_mask=pred_dict['side_output2']
    cv2.imwrite(os.path.join(output_dir,image_basename+'_side2.png'),side2_mask)
    side3_mask=pred_dict['side_output3']
    cv2.imwrite(os.path.join(output_dir,image_basename+'_side3.png'),side3_mask)
    side4_mask=pred_dict['side_output4']
    cv2.imwrite(os.path.join(output_dir,image_basename+'_side4.png'),side4_mask)
    side5_mask=pred_dict['side_output5']
    cv2.imwrite(os.path.join(output_dir,image_basename+'_side5.png'),side5_mask)
    side_fuse_mask=pred_dict['side_output_fuse']
    cv2.imwrite(os.path.join(output_dir,image_basename+'_side_fuse.png'),side_fuse_mask)
    #pdb.set_trace()
    #np.save(os.path.join(output_dir,image_basename+'.npy'),pred_dict['probabilities'])
    #pdb.set_trace()
    mask = Image.fromarray(mask)
    #pdb.set_trace()
    mask.save(path_to_output) 
    #plt.axis('off')
    #plt.imshow(mask)
    #plt.savefig(path_to_output, bbox_inches='tight')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
