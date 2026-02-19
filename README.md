# Installation:

There are 2 ways to install based on your hardware. If you have an Nvidia GPU, use the first option, if not use the second option.

1) Nvidia GPU Enabled Install:

  Use WSL Clone the GitHub
  
    git clone https://github.com/Collinformatics/CleaveNet
  
  Create conda environment:

     conda env create -f environment.yml

  Activate the virtual environment:

    conda activate cleavenet

  Test GPU activation:

    python testGPU.py

  In the future, we can monitor GPU useage with:
  
    watch -n 1 nvidia-smi

 
2) Non-GPU Install:

  Intall Python and use "requirements.txt" to install the python modules

    pip install -r requirements.txt


