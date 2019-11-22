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
with tf.device('/GPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # 2 x 3 
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 3 x 2
  W = tf.matmul(a, b)                                   # 2 x 2

with tf.device('/GPU:1'):
  c = tf.constant([[100.0, 100.0], [100.0, 100.0]])     # 2 x 2
  d = tf.matmul(W, c)                                   # 2 x 2
print('d: {}'.format(d))

with tf.device('/GPU:0'):
  e = d + W                                             # 2 x 2
print('e: {}'.format(e))

# %%
