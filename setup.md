This README explains how to setup and run IBM federated learning from scratch on CIRCE cluster. All commands are assumed to be
run from the base directory at the top of this repository.

## Setup IBM federated learning

To run projects in IBM federated learning, you must first install all the requirements. 
The cluster admnistrator requires users to install Miniconda to work with CIRCE cluster. If you don't have Miniconda,
you can install as shown below. 

Please note: before you start the install directions below, please ensure that you do not have any modules loaded by running the command:
`module purge`

    ```commandline
    cd $HOME
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    #answer "yes" to adding Miniconda3 install location to PATH in your ~/.bashrc
    #log-out of CIRCE, then log back in
    which conda pip
    pip install --upgrade pip
    conda update --all -y
    ```
Next step is to install Anaconda. However, Anaconda installation takes up your home directory's space. Thus, after installation, your home directory siz might exceed the memory or file count quota. You can read more about Anaconda under this [link](https://towardsdatascience.com/managing-project-specific-environments-with-conda-b8b50aa8be0e).

    `
    conda install -y anaconda
    `
Once Conda/Miniconda is installed, you can then create virtual environments for each package (or combination of packages) that you may need, including Tensorflow, Keras, OpenCV, etc.

#### Installation with Conda (recommended)

1. If you already have Conda installed, create a new environment for IBM FL by running:

    `conda create -n <env_name> python=3.6`

    **Note**: Latest IBM FL library supports Keras model training with two different 
    Tensorflow Backend versions(1.15 and 2.1). It is recommended to install IBM FL 
    in different conda environment with different tf versions. I use the latest tensorflow, which is tensorflow 2.1 for this documentation. Thus, there are extra code that we need to add because the current version strongly supports tensorflow 1.15. However, the migration is straightforward because there are sample codes written in tensorflow 1.15.
    
    a. While running Keras experiments with Tensorflow v1.15, create a new environment 
    by running:

        `conda create -n <env_name> python=3.6 tensorflow=1.15`
    
    We provide Tensorflow v1.15 for reference purpose only.

    b. While running Keras experiments with Tensorflow v2.1, try creating a new environment by running:

        `conda create -n <env_name> python=3.6 tensorflow=2.1.0`
 
    **Note**: Tensorflow v2.1 may not be available through conda install. If you get a `PackagesNotFoundError` after running the above command, please try creating a new envirnoment using pip. 
    
    Run `conda create -n <env_name> python=3.6`
    
    After activating the new Conda environment (see Step 2), use `pip install tensorflow==2.1` to install the required tensorflow package.

2. Run `conda activate <env_name>` to activate the new Conda environment.
3. Test if tensorflow 2.0 is installed successfully by running a Python session on terminal. 
4. Install the IBM FL package by running:
    
    `pip install <IBM_federated_learning_whl_file>`


#### Installation with pip

1. Create a virtual environment by running:

    ```commandline
    python -m pip install --user virtualenv
    virtualenv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    ```

    **Then run 'source/venv/bin/activate' to enable the virtual environment.**

2. Install basic dependencies:

    `pip install -r requirements.txt`

3. Install the IBM FL package by running:
    
    `pip install <IBM_federated_learning_whl_file>`


## Split Sample Data

You can use `generate_data.py` to generate sample data on any of the integrated datasets. For example, you could run:
```commandline
python examples/generate_data.py -n 2 -d mnist -pp 200
```

This command would generate 2 parties with 200 data points each from the MNIST dataset. By default
the data is scaled down to range between 0 and 1 and reshaped such that each image is (28, 28). For
more information on what preprocessing was performed, check the [Keras classifier example](/examples/keras_classifier).

Run `python examples/generate_data.py -h` for full descriptions
of the different options. 

## Create Configuration Files

To run IBM federated learning, you must have configuration files for the aggregator and for each party.

You can generate these config files using the `generate_configs.py` script.
 
For example, you could run:

```commandline
python examples/generate_configs.py -f iter_avg -m tf -n 2 -d mnist -p examples/data/mnist/random 
```

This command would generate the configs for the `tensorflow 2.0` model, assuming 2 parties.
You must also specify the party data path via `-p`. 

Run `python examples/generate_configs.py -h` for full descriptions of the different options.
