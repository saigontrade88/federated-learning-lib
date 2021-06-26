# Quickstart - Personal use only

## Try out the step-by-step Tensorflow - Keras classifier example (TF 2.0).
### 0. Acquire Infinity band IP addresses, not the Ethernet IP addres of computing nodes on RC
Use
aggregator:  mdc-1057-30-7 - ib0 ip: 10.250.62.194
party0: mdc-1057-30-8 - ib0 ip: 10.250.62.195
party1:        mdc-1057-30-9 - ib0 ip: 10.250.62.196
For example
```commandline
ping -c 3 mdc-1057-30-7 returns 10.250.46.194 which is the Ethernet IP address, but we use the Infininty band IP: 10.250.(46 + 16).194
```

### 0. Start the aggregator (parameter) server in the first terminal:
Use
```commandline
srun --nodes=1 --ntasks-per-node=1 --mem-per-cpu=4G --time=01:00:00 --partition=devel --qos=devel --nodelist=mdc-1057-30-7 --pty /bin/bash
module purge
conda activate tf_21_cpu
```
Move to the working directory or also refer to as <whl_directory>.
Use 
```commandline
cd /home/l/longdang/Desktop/federated-learning-lib
```
In this example, we will train a Keras CNN model, as shown in figure below, on
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) data in the federated learning fashion. 
```python
num_classes = 10
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```

### 1. Set up a running environment for IBM federated learning.

If you already have Conda installed, create a new conda environment for IBM federated learning by running:
```commandline
conda create -n tf_21 python=3.6 tensorflow-gpu=2.1
```     
Follow the prompts to install all the required packages.

Run `conda activate tf_21` to activate the new Conda environment, and install the IBM federated learning package by running:
```commandline
pip install federated-learning-lib/federated_learning_lib-1.0.5-py3-none-any.whl
```  

**Note**: Lastest IBM FL library supports Keras model training with two different Tensorflow Backend versions (1.15 and 2.1). It is recommended to install IBM FL in different conda environment with different tf versions. See [here](setup.md#installation-with-conda-recommended) for details of how to set up IBM FL with a specific tensorflow backend.

### 2. Prepare datasets for each participating parties.

For example, run
```commandline
python examples/generate_data.py -n 2 -d mnist -pp 200 
```
This command would generate 2 parties with 200 data points each, randomly sampled from the MNIST dataset.
By default, the data is stored under the `examples/data/mnist/random` directory.
```buildoutcfg
Warning: test set and train set contain different labels
Party_ 0
nb_x_train:  (200, 28, 28) nb_x_test:  (5000, 28, 28)
* Label  0  samples:  22
* Label  1  samples:  30
* Label  2  samples:  16
* Label  3  samples:  22
* Label  4  samples:  17
* Label  5  samples:  25
* Label  6  samples:  12
* Label  7  samples:  15
* Label  8  samples:  19
* Label  9  samples:  22
Finished! :) Data saved in  examples/data/mnist/random
Party_ 1
nb_x_train:  (200, 28, 28) nb_x_test:  (5000, 28, 28)
* Label  0  samples:  22
* Label  1  samples:  18
* Label  2  samples:  22
* Label  3  samples:  23
* Label  4  samples:  14
* Label  5  samples:  23
* Label  6  samples:  22
* Label  7  samples:  20
* Label  8  samples:  18
* Label  9  samples:  18
Finished! :) Data saved in  examples/data/mnist/random
```
For a full description of the different options to prepare datasets, run `python examples/generate_data.py -h`.

### 3. Define model specification and create configuration files for the aggregator and parties.

For example, run:
```commandline
python examples/generate_configs.py -n <num_parties> -f iter_avg -m keras -d mnist -p <path>
```
This command performs two tasks:

1) It specifies the machine learning model to be trained, in this case, a Keras CNN classifier.  

2) It generates the configuration files necessary to train a `keras` model via fusion algorithm iter_avg, assuming `<num_parties>` parties join the federated learning training.
You must also specify the dataset name via `-d` and the party data path via `-p`. 

In this example, we run:
```commandline
python  examples/generate_configs.py -n 2 -f iter_avg -m tf -d mnist -p examples/data/mnist/random/
```
Hence, we generate 2 parties in our example, using the `mnist` dataset and `examples/data/mnist/random` as our data path.
```buildoutcfg
Finished generating config file for aggregator. Files can be found in:  <whl_directory>/examples/configs/iter_avg/tf/config_agg.yml
Finished generating config file for parties. Files can be found in:  <whl_directory>/examples/configs/iter_avg/tf/config_party*.yml
```
You may also see warning messages which are fine.
For a full description of the different options, run `python examples/generate_configs.py -h`.

Below you can see samples of configuration files.
- Aggregator's configuration file:
```yaml
connection:
  info:
    ip: 127.0.0.1
    port: 5000
    tls_config:
      enable: false
  name: FlaskConnection
  path: ibmfl.connection.flask_connection
  sync: false
data:
  info:
    npz_file: examples/datasets/mnist.npz
  name: MnistTFDataHandler
  path: ibmfl.util.data_handlers.mnist_keras_data_handler
fusion:
  name: IterAvgFusionHandler
  path: ibmfl.aggregator.fusion.iter_avg_fusion_handler
hyperparams:
  global:
    max_timeout: 120 #Increase this number if necessary. Source code is under examples/generate_configs.py
    num_parties: 2
    rounds: 3
    termination_accuracy: 0.9
  local:
    training:
      epochs: 3
protocol_handler:
  name: ProtoHandler
  path: ibmfl.aggregator.protohandler.proto_handler

```
- Party's configuration file:
```yaml
aggregator:
  ip: 127.0.0.1
  port: 5000
connection:
  info:
    ip: 127.0.0.1
    port: 8085 #You can play with this number if it is necessary. I used 5001 based on [their example] (https://github.com/IBM/federated-learning-lib/blob/main/runner/examples/mnist/config_runner.yml) . I need advice on how to pick this number. 
    tls_config:
      enable: false
  name: FlaskConnection
  path: ibmfl.connection.flask_connection
  sync: false
data:
  info:
    npz_file: examples/data/mnist/random/data_party0.npz
  name: MnistKerasDataHandler
  path: ibmfl.util.data_handlers.mnist_keras_data_handler
local_training:
  name: LocalTrainingHandler
  path: ibmfl.party.training.local_training_handler
model:
  name: TensorFlowFLModel
  path: ibmfl.model.keras_fl_model
  spec:
    model_definition: examples/configs/iter_avg/keras/compiled_keras.h5 # For me, the parties could not load the h5 file. I updated it to be "examples/configs/iter_avg/tf" and it works!!! Need to do experiments on this key.
    model_name: tf-cnn
protocol_handler:
  name: PartyProtocolHandler
  path: ibmfl.party.party_protocol_handler
``` 
Notice that the configuration files contain a `data` section that is different for each party. In fact, each party's points to its own data, generated from the command in step 2.

### 4. Start the aggregator

To start the aggregator, open the terminal window running the IBM federated fearning environment set up beforehand,
and check that you are in the correct directory.  In the terminal run:
```commandline
python -m ibmfl.aggregator.aggregator examples/configs/iter_avg/tf/config_agg.yml 2> stderr_agg.txt | tee stdout_agg.txt
```
where the path provided is the aggregator configuration file path. 
```buildoutcfg
2021-06-11 10:22:28,823 | 1.0.5 | INFO | ibmfl.util.config                             | Getting details from config file.
2021-06-11 10:22:30,993 | 1.0.5 | INFO | ibmfl.util.config                             | No metrics recorder config provided for this setup.
2021-06-11 10:22:30,993 | 1.0.5 | INFO | ibmfl.util.config                             | No model config provided for this setup.
2021-06-11 10:22:31,231 | 1.0.5 | INFO | ibmfl.util.config                             | No data config provided for this setup.
2021-06-11 10:22:31,231 | 1.0.5 | INFO | ibmfl.util.data_handlers.mnist_keras_data_handler | Loaded training data from examples/datasets/mnist.npz
2021-06-11 10:22:31,641 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | RestSender initialized
2021-06-11 10:22:31,641 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.START
2021-06-11 10:22:31,641 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Receiver Initialized
2021-06-11 10:22:31,641 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Initializing Flask application
2021-06-11 10:22:31,645 | 1.0.5 | INFO | __main__                                      | Aggregator initialization successful
```
Then in the terminal, type `START` and press enter.
```buildoutcfg
2021-06-11 10:22:47,630 | 1.0.5 | INFO | root                                          | State: States.CLI_WAIT
2021-06-11 10:22:47,630 | 1.0.5 | INFO | __main__                                      | Aggregator start successful
 * Serving Flask app 'ibmfl.connection.flask_connection' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
2021-06-11 10:22:47,632 | 1.0.5 | INFO | werkzeug                                      |  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

### 5. Start and register parties 

To start and register a new party, open one new terminal window for each party, running the IBM federated learning environment set up beforehand,
and make sure you are in the correct directory. For example, in the terminal run:
```commandline
ssh longdang@svc-3024-9-12
conda activate tf_21
cd ~/Desktop/federated-learning-lib/
python -m ibmfl.party.party examples/configs/iter_avg/keras/config_party<idx>.yml
```
where the path provided is the path to the party's configuration file.

**NOTE**: Each party will have a different configuration file;
in our example, it is noted by changing `config_party<idx>.yml`.
For instance, to start the 1st party, one would run:
```commandline
 python -m ibmfl.party.party examples/configs/iter_avg/tf/config_party0.yml 2>std_err_party0.txt > stdout_party0.txt
```
You may also see warning messages which are fine.
In the terminal for each party, type `START` and press enter to start the party. 
Then type `REGISTER` and press enter to register the party for the federated learning task.
```buildoutcfg
START
REGISTER
```
The aggregator terminal will also prompt out INFO to show that it receives the party's registration message (as shown in the third figure on the right).
```buildoutcfg
2021-06-11 10:23:57,316 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :6
2021-06-11 10:23:57,317 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Adding party with id 44874e3d-b19e-430c-89ca-000d066a79f3
2021-06-11 10:23:57,317 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Total number of registered parties:1
2021-06-11 10:23:57,318 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:23:57] "POST /6 HTTP/1.1" 200 -
2021-06-11 10:24:00,900 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :6
2021-06-11 10:24:00,900 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Adding party with id de2517a4-a3dd-429b-b697-e81dc99f6bd1
2021-06-11 10:24:00,900 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Total number of registered parties:2
2021-06-11 10:24:00,901 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:24:00] "POST /6 HTTP/1.1" 200 -
```

### 6. Initiate training from the aggregator
To initiate federated training, type `TRAIN` in your aggregator terminal and press enter.
**NOTE**: In this example, we have 2 parties join the training and we run 3 global rounds, each round with 3 local epochs.

Outputs in the aggregator terminal after running the above command will look like:
```buildoutcfg
TRAIN
2021-06-11 10:24:24,543 | 1.0.5 | INFO | root                                          | State: States.TRAIN
2021-06-11 10:24:24,543 | 1.0.5 | INFO | __main__                                      | Initiating Global Training.
2021-06-11 10:24:24,544 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_handler        | Warm start disabled.
2021-06-11 10:24:24,544 | 1.0.5 | INFO | ibmfl.aggregator.fusion.iter_avg_fusion_handler | Model updateNone
2021-06-11 10:24:24,544 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.SND_MODEL
2021-06-11 10:24:24,544 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.SND_REQ
2021-06-11 10:24:24,645 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Total number of success responses :2
2021-06-11 10:24:24,645 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Number of parties queried:2
2021-06-11 10:24:24,646 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Number of registered parties:2
2021-06-11 10:24:24,646 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.QUORUM_WAIT
2021-06-11 10:24:25,421 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :7
2021-06-11 10:24:25,524 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :7
2021-06-11 10:24:25,532 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:24:25] "POST /7 HTTP/1.1" 200 -
2021-06-11 10:24:25,644 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:24:25] "POST /7 HTTP/1.1" 200 -
2021-06-11 10:24:29,651 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Timeout:120 Time spent:5
2021-06-11 10:24:29,651 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.PROC_RSP
2021-06-11 10:24:29,652 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.RCV_MODEL
2021-06-11 10:24:29,652 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.AGGREGATING
2021-06-11 10:24:29,681 | 1.0.5 | INFO | ibmfl.aggregator.fusion.iter_avg_fusion_handler | Model update<ibmfl.model.model_update.ModelUpdate object at 0x2b830daccb00>
2021-06-11 10:24:29,682 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.SND_MODEL
2021-06-11 10:24:29,682 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.SND_REQ
2021-06-11 10:24:29,988 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Total number of success responses :2
2021-06-11 10:24:29,988 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Number of parties queried:2
2021-06-11 10:24:29,988 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Number of registered parties:2
2021-06-11 10:24:29,988 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.QUORUM_WAIT
2021-06-11 10:24:30,204 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :7
2021-06-11 10:24:30,209 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :7
2021-06-11 10:24:30,395 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:24:30] "POST /7 HTTP/1.1" 200 -
2021-06-11 10:24:30,400 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:24:30] "POST /7 HTTP/1.1" 200 -
2021-06-11 10:24:34,994 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Timeout:120 Time spent:5
2021-06-11 10:24:34,995 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.PROC_RSP
2021-06-11 10:24:34,995 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.RCV_MODEL
2021-06-11 10:24:34,995 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.AGGREGATING
2021-06-11 10:24:35,021 | 1.0.5 | INFO | ibmfl.aggregator.fusion.iter_avg_fusion_handler | Model update<ibmfl.model.model_update.ModelUpdate object at 0x2b839d412ba8>
2021-06-11 10:24:35,021 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.SND_MODEL
2021-06-11 10:24:35,021 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.SND_REQ
2021-06-11 10:24:35,385 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Total number of success responses :2
2021-06-11 10:24:35,385 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Number of parties queried:2
2021-06-11 10:24:35,385 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Number of registered parties:2
2021-06-11 10:24:35,385 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.QUORUM_WAIT
2021-06-11 10:24:35,516 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :7
2021-06-11 10:24:35,522 | 1.0.5 | INFO | ibmfl.connection.flask_connection             | Request received for path :7
2021-06-11 10:24:35,651 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:24:35] "POST /7 HTTP/1.1" 200 -
2021-06-11 10:24:35,694 | 1.0.5 | INFO | werkzeug                                      | 127.0.0.1 - - [11/Jun/2021 10:24:35] "POST /7 HTTP/1.1" 200 -
2021-06-11 10:24:40,390 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | Timeout:120 Time spent:5
2021-06-11 10:24:40,391 | 1.0.5 | INFO | ibmfl.aggregator.protohandler.proto_handler   | State: States.PROC_RSP
2021-06-11 10:24:40,391 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.RCV_MODEL
2021-06-11 10:24:40,391 | 1.0.5 | INFO | ibmfl.aggregator.fusion.fusion_state_service  | Fusion state States.AGGREGATING
2021-06-11 10:24:40,399 | 1.0.5 | INFO | ibmfl.aggregator.fusion.iter_avg_fusion_handler | Reached maximum global rounds. Finish training :)
2021-06-11 10:24:40,400 | 1.0.5 | INFO | __main__                                      | Finished Global Training
```
Outputs in party's (party 1) terminal after running the above command is stored in the following output files for debugging purposes:
```buildoutcfg
stdout_party0.txt # standard output
std_err_party0.txt # standard error
```
Outputs from party 2 will be similar as party 1.

### 7. (Optional) Issue various commands to train again, evaluate, sync and save the models. 
For a full list of supported commands, see `examples/README.md`.
Sample outputs of issuing the `EVAL` command in one of the parties' terminal after the global training.
```buildoutcfg
stdout_party0.txt # standard output
std_err_party0.txt # standard error
```
Users can also enter `TRAIN` again at the aggregator's terminal if they want to continue the FL training.
Entering `SYNC` at the aggregator's terminal will trigger the synchronization of the current global model with parties, 
and `SAVE` will trigger the parties to save their models at the local working directory.

### 8. Terminate the aggregator and parties processes.
Remember to use `STOP` to terminate the aggregator's and parties' processes and exit. For example,
In the aggregator terminal, run `STOP'.
In the party terminal, run 'STOP'.

 
