# Installation:

These instructions will walk you through installing the program in the terminal.

- Requirements:
    - If you are using a Windows OS, you need to install and use WSL.
    - You need to have conda installed.

Clone the GitHub

    git clone https://github.com/Collinformatics/CleaveNet

Create conda environment:

- If you are using MacOS run:

      conda env create -f environment_mac.yml

- If not, run:

       conda env create -f environment.yml

Activate the virtual environment:

      conda activate cleavenet

Test GPU activation:

      python testGPU.py

If you are using an NVIDIA GPU, you can monitor GPU usage with:

      watch -n 1 nvidia-smi

## Note:

If the line "import cleavenet" gives you an error you'll need to add the working directory to PYTHONPATH:

    export PYTHONPATH="$PYTHONPATH:$PWD"

All training data should be saved in the "data" directory.

# Training:

There are two available models, an LSTM and a Transformer. The Transformer is the default option. To use the LSTM add this flag:

    --model-type lstm

## Generator:

To train a model that can generate protein substrates:

    python src/train_generator.py --data-path <filepath>

Multiple parameters can be adjusted, to print the options run:

    python src/train_generator.py --help


## Predictor:

To train a model that can predict substrate activity:

    python src/train_predictor.py --data-path <filename>

Multiple parameters can be adjusted, to print the options run:

    python src/train_generator.py --help

- The external validation set is optional, if a file is not provided 5% of the dataset will be used for this set. To specify an external validation set use this flag:

      --data-pathEV <filename>




