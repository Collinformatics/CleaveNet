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
	"--model", default=None, type=str, help="file name for the model"
)
parser.add_argument(
	"--num-seqs", default=10, type=int, help="number of sequences to be generated"
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
modelName = 'model_0-Mpro2_Pred_8_AA_Reading_Frame_Q@R4-AUTOREG_transformer-both_rounded.keras'
model = cleavenet.models.load_model(modelName=args.model, seqLen=seqLen)

# 
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


# Generate sequenes
tokenized_seqs = []
untokenized_seqs = []
for i in range(len(conditioning_tag)):
	for j in tqdm(range(args.num_seqs)):
		while True:
			#model.built=True

			# Generate using loaded weights
			generated_seq = cleavenet.models.inference(
				model, dataloader, causal=True, seq_len=seqLen+1, penalty=args.repeat_penalty,
				verbose=False, conditioning_tag=[conditioning_tag[i]], temperature=args.temperature
			)
			tokenized_seqs.append(generated_seq)
			seq = ''.join(dataloader.idx2char[generated_seq])
			if len(seq) == seqLen and '$' not in seq and '*' not in seq:
				untokenized_seqs.append(seq)
				break
				

# Print seqs
print('\nGenerated Substrates:')
for seq in untokenized_seqs:
	print(f'  {seq}')

# Save sequences
if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)
idx = 0
tag = dataset.replace(" - ", " ").replace(" ", "_")
while True:
	save_file = os.path.join(args.output_dir, f'generatedSubs_{idx}-{tag}-penalty_'+str(args.repeat_penalty)+'-temp_'+str(args.temperature)+'.csv')
	if not os.path.exists(save_file):
		break
	idx += 1
print(f'\nSequences saved at: {save_file}')
with open(save_file, 'a') as f:
	for seq in untokenized_seqs:
		f.write(f'{seq}\n')
	f.write('\nModel:\n')
	f.write(f'{args.model}')
       
