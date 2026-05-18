import argparse
import datetime
import math
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm



import cleavenet
from cleavenet import plotter
from cleavenet.utils import get_data_dir


#parse from terminal
parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha", default=0.99, type=float,
    help="Smoothing rate for the exp filter"
)
parser.add_argument(
    "--batch-size", default=128, type=int,
    help="Batch size range"
)
parser.add_argument(
	"--data-path", default="kukreja.csv", type=str,
	help="File path for the training data"
)
parser.add_argument(
	"--data-pathEV", default="Mpro2_ZPred_6AA_ExtValid_MinCounts10.csv",
    type=str, help="File path for the External Validation (EV) data"
)
parser.add_argument(
    "--d-model", default=32, type=int,
    help="Dimensions of model"
)
parser.add_argument(
    "--ensemble", default=5, type=int,
    help="Model iterations to use for ensemble uncertainty calculation"
)
parser.add_argument(
    "--learning-rate", default=0.005, type=float,
    help="Learning rate"
)
parser.add_argument(
    "--log-freq", default=50, type=int,
    help="Frequency to log to tensorboard"
)
parser.add_argument(
    "--max-len", default=10, type=int,
    help="Maximum sequence length to add to dataset"
)
parser.add_argument(
    "--model-type", default='transformer', type=str,
    help="Transformer or lstm architecture"
)
parser.add_argument(
    "--num-epochs", default=50, type=int,
    help="Number of epochs"
)
parser.add_argument(
    "--save-freq", default=100, type=int,
    help="frequency to save weights"
)
parser.add_argument(
    "--split", default=0.8, type=float,
    help="Train val split ratio for ensembling"
)
parser.add_argument(
    "--regu", default=0.01, type=float,
    help=" Regularization parameter for LSTM"
)
args = parser.parse_args()

###################################################################################################
# Get data, split into train and test
###################################################################################################

# Get path to data
data_dir = get_data_dir()
if 'data/' in args.data_path:
    data_path = args.data_path
else:
    data_path = os.path.join('data', args.data_path)
if 'data/' in args.data_pathEV:
    data_pathEV = args.data_pathEV
else:
    data_pathEV = os.path.join('data', args.data_pathEV)
datasetEV = data_pathEV.replace('data/', '').replace('.csv', '')

# Evaluate data_path
if '_' in args.data_path and 'AA' in args.data_path:
    dataset = args.data_path.replace('data/', '').replace('.csv', '')
    sfname = args.data_path.split('_')
    for s in sfname:
        if 'AA' in s:
            args.max_len = int(s.strip('AA'))
else:
    dataset = args.data_path.strip('.csv')
# f'Training Dataset: {dataset}\n'
print(f'\nTraining Model: {args.model_type}\n'
      f'Dataset: {dataset}\n'
      f'Training Data: {data_path}\n'
      f'Dataset Ext Valid: {datasetEV}\n'
      f'Ext Valid Data: {data_pathEV}\n'
      f'Max Length: {args.max_len}\n'
      f'Ensemble: {args.ensemble}\n')

random_seed = list(range(args.ensemble))

###################################################################################################
# Function to run
###################################################################################################
def main():
    if 'kukreja' in dataset:
        from cleavenet.utils import get_data_dir, mmps

        enzCol = mmps
        enzList = ['MMP1', 'MMP10', 'MMP12', 'MMP13', 'MMP17', 'MMP3', 'MMP7']
        enzIdx = []
        for i, m in enumerate(enzCol):
            if m in enzList:
                enzIdx.append(i)

        # Load in kukreja data
        dataloader = cleavenet.data.DataLoader(
            data_path, seed=0, task='regression',
            model=args.model_type, test_split=0.2,
            dataset='kukreja'
        )

        # Load external validation dataset
        dataloaderEV = cleavenet.data.DataLoader(
            data_pathEV, seed=0, task='regression',
            model=args.model_type, test_split=0,
            dataset='bhatia', use_dataloader=dataloader
        )
        xExtValid = cleavenet.data.tokenize_sequences(dataloaderEV.X, dataloader)
        if args.model_type == 'transformer':
            cls_idx = dataloader.char2idx[dataloader.CLS]
            xExtValid = np.stack([np.append(np.array(cls_idx), s) for s in xExtValid])
        yExtValid = dataloaderEV.y
    else:
        enzCol = [dataset.split('_')[0]]
        enzList = enzCol
        enzIdx = [i for i in range(len(enzList))]
        
        dataloader = cleavenet.data.DataLoader(
            data_path, seed=0, task='regression',
            model=args.model_type, test_split=0.2,
            dataset=dataset
        )
        # Load external validation dataset
        dataloaderEV = cleavenet.data.DataLoader(
            data_pathEV, seed=0, task='regression',
            model=args.model_type, test_split=0,
            dataset=datasetEV, use_dataloader=dataloader
        )
        # sys.exit()
        xExtValid = cleavenet.data.tokenize_sequences(dataloaderEV.X, dataloader)
        if args.model_type == 'transformer':
            cls_idx = dataloader.char2idx[dataloader.CLS]
            xExtValid = np.stack([np.append(np.array(cls_idx), s) for s in xExtValid])
        yExtValid = dataloaderEV.y


    N = len(dataloader.y)
    Ntest = len(dataloader.y_test)
    Ntrain = len(dataloader.X_train)
    print(f'\nDataLoader: N={N:,}\n'
          f' y_train: {Ntrain:,}, {100 * round(Ntrain / N, 2)} %\n'
          f'  y_test: {Ntest:,}, {100 * round(Ntest / N, 2)} %\n')

    print(f'External Validation:\n'
          f'  x: {len(xExtValid)}\n'
          f'  y: {len(yExtValid)}\n')
    # print(f'Enzyme Col: {enzCol}\nEnzyme List: {enzList}\nIdx: {enzIdx}\n')
    # sys.exit()

    # Run ensemble training
    init = True
    results = {}
    for ensemble in range(args.ensemble):
        # Train/valid splits for each ensemble, use pre-split data to preserve test set
        X_train, X_valid, y_train, y_valid = train_test_split(
            dataloader.X_train, dataloader.y_train, test_size=1-args.split,
            random_state=random_seed[ensemble]
        )

        vocab_size = len(dataloader.char2idx)
        num_samples = len(X_train)
        num_valid_samples = len(X_valid)
        if init:
            init = False
            print(f'Split Training Set:')
            print(f'* Train:\n'
                  f'    X: {len(X_train)}, {100 * round(len(X_train) / N, 2)} %\n'
                  f'    Y: {len(y_train)}, {100 * round(len(y_train) / N, 2)} %')
            print(f'* Validation:\n'
                  f'    X: {len(X_valid)}, {100 * round(len(X_valid) / N, 2)} %\n'
                  f'    Y: {len(y_valid)}, {100 * round(len(y_valid) / N, 2)} %')
            print(f'vocab size: {vocab_size}\n')
        print(f'Training samples: {num_samples:,}, '
              f'Validation samples: {num_valid_samples:,}')

        run_name = "run-%d" % ensemble
        print(f'--- Starting trial: {run_name}\n')

        # Build the predictor model
        if args.model_type == 'lstm':
            transformer=False
            embedding_dim= 22 # args.d_model
            dropout=0.25
            model = cleavenet.models.RNNPredictor(
                vocab_size, embedding_dim, args.d_model, dropout,
                args.regu, args.max_len, len(enzCol), mask_zero=True
            )
            lr = args.learning_rate # 0.005

        elif args.model_type == 'transformer':
            transformer=True
            num_layers=4
            num_heads=8
            dropout=0.01
            embedding_dim = 128
            model = cleavenet.models.TransformerEncoder(
                num_layers=num_layers,
                d_model=embedding_dim,
                num_heads=num_heads,
                dff=args.d_model,  # dense params
                vocab_size=vocab_size,
                dropout_rate=dropout,
                output_dim=len(enzCol),
                pool_outputs=True,
                mask_zero=True)
            lr = cleavenet.models.TransformerSchedule(args.d_model)

        # print("Learning rate", lr)

        model_label='/'+args.model_type+'_'+str(ensemble)

        model.build((args.batch_size, None))
        model.summary()
        print()

        optimizer = tf.optimizers.Adam(lr)

        # sys.exit()

        @tf.function  # comment out for eager execution (if you want to debug)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                y_hat = model(x, training=True)  # forward pass
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
        running_rmse = None
        best_val_loss = float('inf')
        best_val_loss_path = ''
        best_extVal_loss = float('inf')
        best_extVal_loss_path = ''

        # LOGGING
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        idx = 0
        while True:
            current_time = datetime.datetime.now().strftime("%Y%m%d")
            save_dir = os.path.join(
                'save' + model_label, f'{current_time}-{idx}_PREDICTOR'
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                break
            idx += 1
        train_log_dir = os.path.join(
            'logs' + model_label, f'{current_time}-{idx}_PREDICTOR'
        )
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join(
            'logs' + model_label, f'{current_time}-{idx}_PREDICTOR_val'
        )
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        l = len(str(args.num_epochs))
        for epoch in range(args.num_epochs + 1):
            print(f'Epoch: {epoch} / {args.num_epochs}')
            pbar = tqdm(range(num_samples // args.batch_size))
            for iter in pbar:
                # Grab a batch and train
                x, y = cleavenet.data.get_batch(
                    X_train, y_train, args.batch_size,
                    dataloader, transformer=transformer
                )
                loss, y_hat = train_step(x, y)
                rmse = model.compute_rmse(y, y_hat)  # compute train rmse

                running_loss = smooth(running_loss, loss.numpy())
                running_rmse = smooth(running_rmse, rmse.numpy())

                global_step += 1

                # saving
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=global_step)
                    tf.summary.scalar('rmse', rmse, step=global_step)

            print(2 * '\033[F\033[K', end='')  # Clear progress bar
            if epoch > 0:  # run validation every epoch
                # print("Running validation")
                vbar = tqdm(range(math.ceil(len(X_valid) / args.batch_size)))
                val_loss = []
                val_rmse = []
                for v_iter in vbar:
                    xv, yv = cleavenet.data.get_batch(
                        X_valid, y_valid, args.batch_size,
                        dataloader, transformer=transformer
                    )
                    yv_hat = model(xv, training=False)
                    val_loss.append(model.compute_loss(yv, yv_hat)*args.batch_size) # compute loss
                    val_rmse.append(model.compute_rmse(yv, yv_hat)*args.batch_size) # compute val rmse
                val_loss = np.sum(val_loss)/len(X_valid) # batch-averaged loss
                val_rmse = np.sum(val_rmse)/len(X_valid)

                # saving
                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=epoch)
                    tf.summary.scalar('rmse', val_rmse, step=epoch)

                    # save weights only if validation loss decreases
                    # print("best val loss:", best_val_loss)
                    if val_loss < best_val_loss:
                        # print(f"Saving with val loss: {val_loss:.4f}")
                        # print(f"Val rmse: {val_rmse:.4f}")
                        best_val_loss_path = os.path.join(
                            save_dir, "{}.weights.h5".format("model")
                        )
                        model.save_weights(best_val_loss_path)
                        best_val_loss = val_loss

                ## run external validation
                ext_yv_hat = model(xExtValid, training=False)
                ext_yv_hat_condensed = tf.concat(
                    [tf.expand_dims(ext_yv_hat[:,index], axis=1) for index in enzIdx],
                    axis=1
                )
                extVal_loss = model.compute_loss(
                    yExtValid[:, :len(enzIdx)], ext_yv_hat_condensed
                ) # compute loss
                extVal_rmse = model.compute_rmse(
                    yExtValid[:, :len(enzIdx)], ext_yv_hat_condensed
                )
                print(1 * '\033[F\033[K', end='')  # Clear progress bar

                # Log data
                if epoch % 5 == 0:
                    print(
                        f'Epoch: {epoch}{(l - len(str(epoch))) * " "} | '
                        f'Best Loss: {best_val_loss:.3f} | '
                        f'Loss: {val_loss:.3f} | '
                        f'Val RMSE: {val_rmse:.3f} | '
                        f'Ext Val loss: {extVal_loss.numpy():.3f}'
                    )
                # print("External validation loss:", extVal_loss)
                # saving
                with val_summary_writer.as_default():
                    tf.summary.scalar('b-loss', extVal_loss, step=epoch)
                    tf.summary.scalar('b-rmse', extVal_rmse, step=epoch)

                    # save weights only if validation loss decreases
                    # print("best val loss:", best_extVal_loss)
                    # sys.exit()
                    if extVal_loss < best_extVal_loss:
                        # print(f"Saving with ext valid loss: {extVal_loss:.4f}")
                        # print(f"External valid rmse: {extVal_rmse}")
                        best_extVal_loss_path = os.path.join(
                            save_dir,
                            "{}.weights.h5".format(f"best-{dataset}-model")
                        )
                        model.save_weights(best_val_loss_path)
                        best_extVal_loss = extVal_loss
        save_file = save_dir + '/best_loss.csv'
        with open(save_file, 'w') as f:
            f.write(str(best_val_loss))


        ##################################
        # After training assess performance of trained model in full set of test data
        # using load model here so we can use the best checkpoint
        ensembleStr = str(ensemble)
        ensemble_dir = save_dir
        checkpoint_path_final = os.path.join(ensemble_dir, "model.weights.h5")
        results[ensembleStr] = {}
        results[ensembleStr]['Weights'] = checkpoint_path_final

        # Re-build the predictor model
        if args.model_type == 'lstm':
            model = cleavenet.models.RNNPredictor(
                vocab_size, embedding_dim, args.d_model,
                dropout, args.regu, args.max_len, len(enzCol)
            )
        elif args.model_type == 'transformer':
            model = cleavenet.models.TransformerEncoder(
                num_layers=num_layers,
                d_model=embedding_dim,
                num_heads=num_heads,
                dff=args.d_model,  # dense params
                vocab_size=vocab_size,
                dropout_rate=dropout,
                output_dim=len(enzCol),
                pool_outputs=True
            )

        model.build((len(dataloader.X_test), None))
        model.summary()
        model.load_weights(checkpoint_path_final)  # load weights from best checkpoint
        print(f'Loaded Weights: {checkpoint_path_final}')


        xt, yt = cleavenet.data.get_batch(
            dataloader.X_test, dataloader.y_test, len(dataloader.X_test), dataloader,
            test=True, transformer=transformer
        )
        yt_hat = model(xt, training=False)  # forward pass
        embeddings = model.last_layer_embeddings
        test_rmse = model.compute_rmse(yt, yt_hat, axis=0)  # compute val rmse
        print(f'RMSE:\n'
              f'* {enzList}\n'
              f'* {test_rmse}\n')
        results[ensembleStr][' Enz'] = enzList
        results[ensembleStr]['RMSE'] = test_rmse

        # Save embeddings for later
        np.save(
            os.path.join(ensemble_dir, 'test_weighted_cluster_embedngsdi.npy'),
            np.array(embeddings)
        )

        # Plot results
        # Scatterplot of predicted vs true
        plotter.plot_parity(yt, yt_hat, enzCol, ensemble_dir)
        # plot RMSE of all MMP families
        plotter.plot_rmse(test_rmse, enzCol, ensemble_dir)

    # Print results
    print('Training Results:')
    for ens, data in results.items():
        print(f'* Ensemble: {ens}')
        for k, v in data.items():
            print(f'    {k}: {v}')
        print()


if __name__ == "__main__":
    main()
