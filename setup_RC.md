This README explains how to setup and run IBM federated learning from scratch on RC. 

## Clone IBM federated learning
Use
```commandline
cd /work_bgfs/l/longdang/
git clone https://github.com/IBM/federated-learning-lib.git
pwd
cd federated-learning-lib
'''
All commands are assumed to be run from the current directory.

## Setup IBM federated learning

To run projects in IBM federated learning, you must first install all the requirements. 
We highly recommend using Conda installation for this project. If you don't have Conda,
you can [install it here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

#### Installation with Conda (recommended)

1. If you already have Conda installed, create a new environment for IBM FL by running:

    `conda create -n <env_name> python=3.6`

    **Note**: Latest IBM FL library supports Keras model training with two different 
    Tensorflow Backend versions(1.15 and 2.1). It is recommended to install IBM FL 
    in different conda environment with different tf versions.
    
    a. While running Keras experiments with Tensorflow v1.15, create a new environment 
    by running:

        `conda create -n tf_15_cpu python=3.6 tensorflow=1.15`

    b. While running Keras experiments with Tensorflow v2.1, try creating a new environment by running:

        `conda create -n tf_21_cpu python=3.6 tensorflow=2.1.0 -y`
        
    c. Tensorflow v.21 with GPU support
    
        `conda create -n tf_21_cpu python=3.6 tensorflow-gpu=2.1.0 -y`
     
2. Run `conda activate <env_name>` to activate the new Conda environment. For example:
        `conda activate tf_21_cpu`

3. Install the IBM FL package by running:
    
    `pip install <IBM_federated_learning_whl_file>`
   
   Use
   ```commandline
   pip install federated-learning-lib/federated_learning_lib-1.0.5-py3-none-any.whl 
   '''
 4. Install additional packages:
 ```commandline
   conda install -c anaconda paramiko -y
   pip install http://github.com/IBM/pycloudmessenger/archive/v0.7.1.tar.gz
 '''
