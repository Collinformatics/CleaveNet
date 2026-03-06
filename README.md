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


## Generator:

Training:

- Train a model that can predice protein substrates.

  Multiple parameters can be adjusted, to print the options run:

      python src/train_generator.py --help

- You can train the generator with:

      python src/train_generator.py --data-path <filepath>







