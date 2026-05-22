import argparse
import cleavenet
import os
import pandas as pd
import sys


# Parse terminal inputs
parser = argparse.ArgumentParser()
parser.add_argument(
	"--model-type", type=str, default='transformer',
	help="'transformer' or 'lstm, for most use cases the default should be used'"
)
parser.add_argument(
	"--model-dir", type=str, default=None,
	help="Directory with the date and index where \"model.keras\" are found. "
		 "Ex: save/transformer_N/<model-dir>/model.keras"
)
parser.add_argument(
	"--no-csv-header", action='store_true',
	help="If using data splits from kukreja as described in the README, "
		 "we store the MMP headers separately. Use this flag to indicate that"
)
parser.add_argument(
	"--path-to-sequence-csv", type=str, default='/data/',
	help="Path to csv file where each line should be a new peptide sequence"
)
parser.add_argument(
	"--path-to-zscores", type=str, default=None,
	help="If you want to measure our model predictions against your data, "
		 "provide a path to csv file where each line should be a corresponding z-score, "
		 "see 'splits/y_all.csv' for an example. If using different MMPS, "
		 "the first row of this file should correspond to the MMPs in each row. "
		 "The default is to assign each column to MMPs in the order we analyzed the data"
)
parser.add_argument(
	"--save-dir", type=str, default='outputs/',
	help="Directory to save model outputs too"
)
args = parser.parse_args()

# Format weights dir
if 'PREDICTOR' not in args.model_dir:
	args.model_dir += '_PREDICTOR'
print(f'Model type: {args.model_type}')
print(f'Model dir:  {args.model_dir}')

# Make save dir
if not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir, exist_ok=True)

# Define column names
dataset = None
if 'kukreja' in args.path_to_sequence_csv:
	from cleavenet.utils import mmps
	dataset = 'kukreja'
	enzymes = mmps
else:
	from pathlib import Path
	parts = Path(args.path_to_sequence_csv).parts
	enzymes = [parts[0]]
	for part in parts:
		if '_' in part:
			dataset = part.split('_')[0]
			enzymes = [dataset]
			break
print(f'Enzymes: {enzymes}')
true_scores=None
if args.path_to_zscores is not None:
	if args.no_csv_header:
		true_scores = pd.read_csv(args.path_to_zscores, names=enzymes).to_numpy()
	else:
		true_scores = pd.read_csv(args.path_to_zscores)
		enzymes = true_scores.columns.to_list()
		true_scores = true_scores.to_numpy()
data_path = args.path_to_sequence_csv
if not data_path.startswith('predict'):
	data_path = os.path.join('predict', data_path.lstrip('/'))
print(f'\nPrediction Seqs: {data_path}')

# Load prediction sequences
input_df = pd.read_csv(data_path).set_index('sequence')
eval_sequences = input_df.index.to_list()
print(f'* {", ".join(eval_sequences)}\n')

# Load in dataloader
dataloader = cleavenet.data.DataLoader(
	data_path, seed=0, model='autoreg', test_split=0.2, dataset=dataset
)

pred_zscores, std_zscores = cleavenet.models.prediction(
	data_path, eval_sequences, args.save_dir, dataset=dataset,
	model_weights=args.model_dir, predictor_model_type=args.model_type,
	true_zscores=true_scores, enzymes=enzymes
)

