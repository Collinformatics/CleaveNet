import tensorflow as tf

print(f'TensorFlow: {tf}')
print(f'TensorFlow File: {tf.__file__}')

gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow Version: {tf.__version__}')
print(f'GPU Found: {gpus}')

from tensorflow.python.client import device_lib
print(f'\nDevices:\n{device_lib.list_local_devices()}\n')

if gpus:
    print("GPU(s) detected:")
    for gpu in gpus:
        print(" ", gpu)
    device = "GPU"
else:
    device = "CPU"
print(f'Training Device: {device}\n')

import os

for k, v in os.environ.items():
    if "CUDA" in k:
        print(f"{k} = {v}")
