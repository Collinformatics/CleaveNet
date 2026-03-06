# Installation:

These instructions will walk you through installing the program in the terminal.

- Note: If you are using a Windows OS, you need to install and use WSL.

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

In the future, we can monitor GPU usage with:

  watch -n 1 nvidia-smi



