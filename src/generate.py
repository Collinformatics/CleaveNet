import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


import cleavenet
from cleavenet.data import custom_round
from cleavenet.utils import mmps

parser = argparse.ArgumentParser()
parser.add_argument(
	"--data_path", default="kukreja.csv", type=str, help="file path for the training data"
)
parser.add_argument(
	"--model_weights", default="model_0", type=str, help="file name for the model weights"
)
parser.add_argument(
	"--num-seqs", default=100, type=int, help="number of sequences to be generated"
)
parser.add_argument(
	"--output-dir", default="generated", type=str, help="Directory to store outputs "
)
parser.add_argument(
	"--repeat-penalty", default=1.0, type=float, help="Repeat penalty factor, for no penalty use 1"
)
parser.add_argument(
	"--temperature", default=1.0, type=float, help="Sampling temperature, for standard sampling use 1. For less diverse use 0.7, for more diverse use >1"
)
parser.add_argument(
	"--z-scores", default=None, type=str, help="File containing z-scores for conditional generation"
)
args = parser.parse_args()

tf.config.list_physical_devices('GPU')


import sys


# Load in dataloader
#data_dir = cleavenet.utils.get_data_dir()
#data_path = os.path.join(data_dir, "kukreja.csv")

if 'data/' in args.data_path:
	data_path = args.data_path
else:
	data_path = os.path.join('data', args.data_path)

# Evaluate data_path
seqLen = None
if ' - ' in args.data_path and ' AA' in args.data_path:
    dataset = args.data_path.split(' - ')[0].replace('data/', '')
    fname = args.data_path.split(' - ')
    for s in fname:
    	if ' AA' in s:
    		seqLen = int(s.strip(' AA'))
else:
    dataset = args.data_path.strip('.csv')
    seqLen = 10
# f'Training Dataset: {dataset}\n'
print(f'\nTraining Data: {data_path}\n'
      #f'      Dataset: {dataset}\n'
      f' Sequence Len: {seqLen}\n')


dataloader = cleavenet.data.DataLoader(
	data_path, seed=0, task='generator', model='autoreg', test_split=0.2, dataset=dataset
)

# From dataloader get necessary variables
start_id = dataloader.char2idx[dataloader.START]
vocab_size = len(dataloader.char2idx)

# Load model
model, checkpoint_path = cleavenet.models.load_generator_model(
	model_type='transformer', model_weights=args.model_weights, training_scheme='rounded'
)

if 2 == 3:
	x = 3
# Match training seq_len
dummy_seq = tf.zeros((1, seqLen), dtype=tf.int32) # batch_size, seq_len
dummy_cond = tf.zeros((1, 18), dtype=tf.int32)    # conditioning vector
_ = model((dummy_seq, dummy_cond), training=False)
model.summary()

# Now weights can be loaded safely
#model.load_weights(checkpoint_path)

if 2 == 3:
	x = 3
	# Fake run to load data in model (have to do this for conditional models since run in eager mode)
	conditioning_tag_fake = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	print(f'\nCond Tag: ({len(conditioning_tag_fake)}, {len(conditioning_tag_fake[0])})\n')
	generated_seq = cleavenet.models.inference(model, dataloader, causal=True, seq_len=seqLen+1,
			                                   penalty=1, # no penalty
			                                   verbose=True,
			                                   conditioning_tag=conditioning_tag_fake,
			                                   temperature=1 # no temp
			                                   )
	model.summary()

if args.z_scores is not None:
	cond_z_scores = pd.read_csv(args.z_scores)
	assert ([mmp in cond_z_scores.columns for mmp in mmps])
	cond_z_scores = cond_z_scores[mmps]
	for mmp in mmps:
		print(f'MMP: {mmp}')
		cond_z_scores[mmp] = cond_z_scores[mmp].apply(lambda x: custom_round(x, base=0.1))  # round to nearest 0.1
	conditioning_tag = cond_z_scores.values.tolist()
else:
	conditioning_tag = [[start_id]] # unconditional generation


tokenized_seqs = []
untokenized_seqs = []


print('\n')
for layer in model.layers:
	if layer.weights:
		print(layer.name, "has", len(layer.weights), "weights")
	else:
		print(layer.name, "has NO weights!")
print()
#sys.exit()


def print_dense_layers(layer, prefix=""):
    try:
        sublayers = layer.layers  # most layers have this
    except AttributeError:
        sublayers = []
    for sub in sublayers:
        print_dense_layers(sub, prefix + layer.name + "/")
    if isinstance(layer, tf.keras.layers.Dense):
        print(prefix + layer.name)

# Start from top-level model
print('Layers:')
for layer in model.layers:
    print_dense_layers(layer)


import h5py

with h5py.File(checkpoint_path, "r") as f:
    print("\nTop-level groups in weights file:")
    for k in f.keys():
        print(' ', k)
print()


for i in range(len(conditioning_tag)):
	for j in tqdm(range(args.num_seqs)):
		model.built=True
		with h5py.File(checkpoint_path, "r") as f:
			dense_w = f["final_layer/dense/kernel:0"][:]
			dense_b = f["final_layer/dense/bias:0"][:]

			model.get_layer("dense").set_weights([dense_w, dense_b])
		model.load_weights(checkpoint_path)  # Load model weights
		#model.load_weights(checkpoint_path, by_name=True, skip_mismatch=False)
		#print(1)
		#sys.exit()
		
		# Generate using loaded weights
		generated_seq = cleavenet.models.inference(
			model, dataloader, causal=True, seq_len=seqLen+1, penalty=args.repeat_penalty,
			verbose=False, conditioning_tag=[conditioning_tag[i]], temperature=args.temperature
		)
		tokenized_seqs.append(generated_seq)
		untokenized_seqs.append(''.join(dataloader.idx2char[generated_seq]))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

save_file = os.path.join(args.output_dir, 'generated_samples_penalty_'+str(args.repeat_penalty)+'_temp_'+str(args.temperature)+'.csv')
with open(save_file, 'a') as f:
    for seq in untokenized_seqs:
        f.write(seq)
        f.write('\n')
