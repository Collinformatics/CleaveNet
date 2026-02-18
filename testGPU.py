import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU(s) detected:")
    for gpu in gpus:
        print(" ", gpu)
    device = "GPU"
else:
    device = "CPU"
print(f'Training Device: {device}\n')
