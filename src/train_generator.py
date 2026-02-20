import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import cleavenet
from cleavenet.utils import get_data_dir

# Suppress tensorflow message about gpu detection
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Comment out to inspect detection)

# parse from terminal
parser = argparse.ArgumentParser() ##
parser.add_argument(
	"--alpha", default=0.99, type=float, help="smoothing rate for the exp filter"
)
parser.add_argument(
	"--batch-size", default=128, type=int, help="batch size"
)
parser.add_argument(
	"--condition", default='randomize', type=str, help="`unconditional` for unconditional generation, `conditional` for conditional training with z-scores only, `randomize`, enables training of both schemes at 50%"
)
parser.add_argument(
	"--data_path", default='kukreja.csv', type=str, help="file path for the training data"
)
parser.add_argument(
	"--learning-rate", default=0.001, type=float, help="learning rate for LSTM"
)
parser.add_argument(
	"--model-type", default='transformer', type=str, help="transformer or LSTM"
)
parser.add_argument(
	"--d-model", default=64, type=int, help="model paramters"
)
parser.add_argument(
	"--num-epochs", default=300, type=int, help="number of epochs"
)
parser.add_argument(
	"--seq-len", default=10, type=int, help="input sequence length to split training text into"
)
parser.add_argument(
	"--random-seed", default=0, type=int, help="random seed"
)
parser.add_argument(
	"--training-scheme", default='autoreg', type=str, help="`autoreg` for autoregressive training, `bert` for Bert MLM training"
)


args = parser.parse_args()
args.model_type = args.model_type.lower()

rng = random.Random(args.random_seed) # set random seed

# Get path to data
data_dir = get_data_dir()
if 'data/' in args.data_path:
	data_path = args.data_path
else:
	data_path = os.path.join('data', args.data_path)

# Evaluate data_path
if ' - ' in args.data_path and ' AA' in args.data_path:
    dataset = args.data_path.split(' - ')[0]
    fname = args.data_path.split(' - ')
    for s in fname:
    	if ' AA' in s:
    		args.seq_len = int(s.strip(' AA'))
else:
    dataset = args.data_path.strip('.csv')
# f'Training Dataset: {dataset}\n'
print(f'Training Model: {args.model_type}\n'
      f'Training Data: {data_path}\n'
      f'Sequence Length: {args.seq_len}\n'
      f'Training Scheme: {args.training_scheme}\n')

import sys

def main():
	if args.training_scheme == 'autoreg':
		causal=True
		model_label = '/AUTOREG_'+args.model_type
		args.seq_len+=1 # account for start token
		dataloader = cleavenet.data.DataLoader(data_path, seed=0, task='generator', model='autoreg', test_split=0.2, dataset=dataset, rounded=True)
		start_id = dataloader.char2idx[dataloader.START]
		end_id = dataloader.char2idx[dataloader.STOP]   
		print("start_id", start_id)
		print("stop_id", end_id)
	elif args.training_scheme == 'bert':
		causal=False
		model_label='/BERT_'+args.model_type
		dataloader = cleavenet.data.DataLoader(data_path, seed=0, task='generator', model='bert', test_split=0.2, dataset=dataset)
		masking_id = dataloader.char2idx[dataloader.MASK]
		print("masking_id", masking_id)
	else:
		raise ValueError("Unknown training scheme")

	# Get vocabulary size and num samples
	print(dataloader.char2idx)
	vocab_size = len(dataloader.char2idx)
	print("vocab length",  len(dataloader.char2idx))
	print("vocab size",  vocab_size)
	print("Train samples", len(dataloader.X_train), "Test samples", len(dataloader.X_test))

	if args.condition == 'conditional' or args.condition == 'randomize':
		conditioning_tag=dataloader.y_train
		conditioning_tag_test=dataloader.y_test
	else:
		conditioning_tag=None
		conditioning_tag_test=None

	randomize_tag = False
	if args.condition == 'randomize':
		randomize_tag = True


	if args.model_type == 'transformer':
		num_layers = 3
		#num_layers = 2
		num_heads = 6
		dropout = 0.25
		if causal:
		    if args.condition == 'conditional' or args.condition == 'randomize':
		        model = cleavenet.models.ConditionalTransformerDecoder(
		                            num_layers=num_layers,
		                            d_model=args.d_model,
		                            num_heads=num_heads,
		                            dff=args.d_model, # dense params
		                            vocab_size=vocab_size,
		                            dropout_rate=dropout)
		    elif args.condition == 'unconditional':
		        model = cleavenet.models.TransformerDecoder(
		                                num_layers=num_layers,
		                                d_model=args.d_model,
		                                num_heads=num_heads,
		                                dff=args.d_model, # dense params
		                                vocab_size=vocab_size,
		                                dropout_rate=dropout)
		    else:
		        raise ValueError("Unknown model type")
		else:
		    model = cleavenet.models.TransformerEncoder(
		        num_layers=num_layers,
		        d_model=args.d_model,
		        num_heads=num_heads,
		        dff=args.d_model,  # dense params
		        vocab_size=vocab_size,
		        dropout_rate=dropout,
		        mask_zero=False)

		lr = cleavenet.models.TransformerSchedule(args.d_model)

	elif args.model_type == 'lstm':
		regu = 0.01
		if causal:
		    num_layers = 3
		    args.batch_size = 128
		    dropout = 0.2
		    embedding_dim = 64
		    #args.d_model = 32
		    model = cleavenet.models.AutoregressiveRNN(args.batch_size, vocab_size, embedding_dim, args.d_model, dropout,
		                                  regu, args.seq_len, training=True, mask_zero=False, num_layers=num_layers)
		else:
		    args.batch_size = 128
		    dropout = 0.01
		    embedding_dim = 64
		    #args.d_model = 64
		    num_layers = 3
		    args.learning_rate = 0.001
		    model = cleavenet.models.RNNGenerator(args.batch_size, vocab_size, embedding_dim, args.d_model, dropout,
		                                  regu, args.seq_len, training=True, num_layers=num_layers)
		#model.build((args.batch_size, None))
		model.summary()
		lr = args.learning_rate

	optimizer = tf.optimizers.Adam(lr)

	#@tf.function  # comment out for eager execution (if you want to debug)
	def train_step_mask(x, y, mask):
		with tf.GradientTape() as tape:
		    if args.model_type == 'lstm':
		        model.reset_states()
		    y_hat = model(x, training=True)  # forward pass
		    loss = model.compute_masked_loss(y, y_hat, mask)  # compute loss
		    grads = tape.gradient(loss, model.trainable_variables)  # compute gradient
		    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # update
		    return loss, y_hat

	#@tf.function
	def train_step_autoreg(x, y):
		#print(x)
		with tf.GradientTape() as tape:
		    if args.model_type == 'lstm':
		        model.reset_states()
		    y_hat = model(x, training=True)  # forward pass
		    #print("y_hat", y_hat)
		    loss = model.compute_loss(y, y_hat)  # compute loss
		    grads = tape.gradient(loss, model.trainable_variables)  # compute gradient
		    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # update
		    return loss, y_hat

	def smooth(prev, val):
		if prev is not None:
		    new = (1 - args.alpha) * val + args.alpha * prev
		else:
		    new = val
		return new

	global_step = 0
	running_loss = None
	best_val_loss = float('inf')
	n_tokens = 0

	# LOGGING
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	save_dir = os.path.join('save'+model_label, '{}_GEN'.format(current_time))
	os.makedirs(save_dir)
	train_log_dir = os.path.join('logs'+model_label, '{}_GEN_train'.format(current_time))
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	val_log_dir = os.path.join('logs'+model_label, '{}_GEN_val'.format(current_time))
	val_summary_writer = tf.summary.create_file_writer(val_log_dir)
	
	# Model path
	#print(f'\nModel label: {model_label}\nCondition: {args.condition}\nScheme: {args.training_scheme}')
	pathModel = os.path.join(os.getcwd(), 'weights') 
	if args.condition == 'randomize':
		cond = 'both'
		if args.training_scheme == 'autoreg':
			cond += '_rounded'
	else:
		cond = args.condition
	pathDir = os.path.join(pathModel, model_label[1:], cond)
	if not os.path.exists(pathDir):
		os.makedirs(pathDir)
	while True:
		idx = 0
		pathModelWeights = os.path.join(pathDir, f'model_{idx}.weights.h5')
		if not os.path.exists(pathModelWeights):
			pathModelLoss = os.path.join(pathDir, f'model_{idx}_loss.txt')
			break
		idx += 1
	print(f'\nSaving the best model weights at: {pathModelWeights}\n')


	# Train generator
	bestEpoch = ''
	timeStart = time.time()
	for epoch in range(args.num_epochs):
		print("epoch", epoch)
		pbar = tqdm(range(int(len(dataloader.X_train) // args.batch_size)))
		for iter in pbar:
		    # Grab a batch and train
		    if causal:
		        x, y = cleavenet.data.get_autoreg_batch(dataloader.X_train, args.batch_size, dataloader, width=args.seq_len, conditioning_tag=conditioning_tag, rng=rng, randomize_tag=randomize_tag)
		        loss, y_hat = train_step_autoreg(x, y)
		        acc = model.compute_accuracy(y, y_hat)
		        if iter == 0 and epoch == 0:
		            model.summary()
		    else:
		        x, y, mask = cleavenet.data.get_masked_batch(dataloader.X_train, args.batch_size, rng, dataloader)
		        n_tokens += mask.sum()
		        loss, y_hat = train_step_mask(x, y, mask)
		        acc = model.compute_masked_accuracy(y, y_hat, mask)
		    if iter == 0 and epoch == 0:
		        model.summary()
		    running_loss = smooth(running_loss, loss.numpy())

		    # saving
		    with train_summary_writer.as_default():
		        tf.summary.scalar('loss', loss, step=global_step)
		        tf.summary.scalar('accuracy', acc, step=global_step)

		    global_step += 1

		if epoch > 0: # run validation every epoch
			vbar = tqdm(range(len(dataloader.X_test) // args.batch_size))
			val_loss = []
			val_acc = []
			val_tokens = []
			for v_iter in vbar:
				if causal:
					xv, yv = cleavenet.data.get_autoreg_batch(dataloader.X_test, args.batch_size, dataloader, width=args.seq_len, conditioning_tag=conditioning_tag_test, rng=rng, randomize_tag=randomize_tag)
					if args.model_type == 'lstm':
						model.reset_states()
					yv_hat = model(xv, training=False)  # forward pass
					val_loss.append(model.compute_loss(yv, yv_hat)*args.batch_size)
					val_acc.append(model.compute_accuracy(yv, yv_hat)*args.batch_size)
				else:
					xv, yv, mask_v = cleavenet.data.get_masked_batch(dataloader.X_test, args.batch_size, rng, dataloader)
					if args.model_type == 'lstm':
						model.reset_states()
					n_tokens = np.sum(mask_v)
					val_tokens.append(n_tokens)
					yv_hat = model(xv, training=False)  # forward pass
					val_loss.append(model.compute_masked_loss(yv, yv_hat, mask_v)*n_tokens) # compute loss
					val_acc.append(model.compute_masked_accuracy(yv, yv_hat, mask_v)*n_tokens)
			if causal:
				val_loss = np.sum(val_loss)/len(dataloader.X_test)
				val_acc = np.sum(val_acc)/len(dataloader.X_test)
			else:
				val_loss = np.sum(val_loss)/np.sum(val_tokens)
				val_acc = np.sum(val_acc)/np.sum(val_tokens)


			# saving
			with val_summary_writer.as_default():
				tf.summary.scalar('loss', val_loss, step=epoch)
				tf.summary.scalar('accuracy', val_acc, step=epoch)

			# save model and weights only if validation loss decreases
			if val_loss < best_val_loss: 
				print(f"Saving with val loss: {val_loss:.4f}")
				print(f"Val acc: {val_acc:.4f}")
				pathWeights = os.path.join(
					save_dir, "{}_epoch_{}.weights.h5".format("model", epoch)
				)
				model.save_weights(pathWeights)
				best_val_loss = val_loss
				
				# Save best weights in weights dir
				model.save_weights(pathModelWeights)
				with open(pathModelLoss, 'w') as file:
					file.write(str(best_val_loss))


	timeEnd = time.time()
	timeTrain = timeEnd - timeStart
	timeItr = round((args.num_epochs / timeTrain), 2)
	timeTrain = round((timeTrain / 60), 2)
	print(f'Best model weights saved at: {pathModelWeights}')
	print(f'Loss: {best_val_loss}')
	print(f'Training Time: {timeTrain:,}min, {timeItr}epoch/s')
	save_file = save_dir + '/best_loss.csv'
	with open(save_file, 'w') as f:
		f.write(str(best_val_loss))



if __name__ == "__main__":
    main()
