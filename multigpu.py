# %%
import tensorflow as tf
import numpy as np

tf.debugging.set_log_device_placement(True)

# %%
use_sim_gpu = False
if use_sim_gpu:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Create 2 virtual GPUs with 1GB memory each
    try:
      tf.config.experimental.set_virtual_device_configuration(
          gpus[0],
          [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Virtual devices must be set before GPUs have been initialized
      print(e)

# %%
# Create tensors on each GPU
with tf.device('/GPU:0'):
  W = tf.Variable([[1.0, 2.0], [-1.0, -1.0], [-1.0, 2.0]]) # 3 x 2
with tf.device('/GPU:1'):
  b = tf.Variable([[1.0], [1.0], [1.0]])                # 3 x 1
with tf.device('/GPU:0'):
  c = tf.constant([[2.0], [2.0], [2.0]])                   # 3 x 1

# A simple tensor operation
# @tf.function decorator allows better performance using AutoGraph
@tf.function
def simple_op(x):
# input a 2 x 1 array
  with tf.device('/GPU:0'):
    z = tf.matmul(W, x) + b                             # 3 x 1             
  return z

# Use GradientTape to record operations
# Operations in simple_op() will be done on GPU:0 as specified in the def above
# Sigmoid and adding of c are done on GPU:1
with tf.GradientTape() as tape:
  with tf.device('/GPU:1'):
    z = simple_op([[1.0], [-1.0]])                     # 3 x 1
    d = tf.sigmoid(z) + c                              # 3 x 1

# Compute gradients of d w.r.t. W and b on GPU:1
with tf.device('GPU:0'):
  grads = tape.gradient(d, [W, b])

# %%
print('z: {}'.format(z))
print('d: {}'.format(d))
print('dd_dW: {}'.format(grads[0].numpy()))
print('dd_db: {}'.format(grads[1].numpy()))