# little RNN

This repository is the public/restricted version of the private repository little_RNN. I conducted this project at the Swiss laboratory .NeuroRestore in Lausanne in the Cyberspace team for my 6-month master's thesis. The project aims at creating a RNN, trained by backpropagation, that can reproduce the link between the time series of activation of 16 electrodes of the paddle lead .NeuroRestore implements in paraplegic patients, and the time series of activation of 14 muscles of the lower limb (emgs). This model could be use for scientific understanding of the spinal circuits, as a simulation model for new data, or an optimization model while adding a controller.

The report and defense remain confidential because they expose medical data analysis.

In the folder plot_examples, you can take a look at some resulting plots of the main_rnn_training.py script. There was many more but I only chose those that do not compromise medical data.

The source folder (src) is composed of three packages, one intended for data processing and analysis (data), one for the training of the rnn (rnn) and finally one for the training of the controller (controller).

## Workspace setup

1. create conda environment:
   `bash
conda create -f environment.yml
`

2. install specific pytorch tools if you have a GPU (see https://pytorch.org/get-started/locally/)

## Data

Data description and format is restricted.

## Training an RNN

To train a RNN :

1. Check the constant parameters in the **init**.py file of each package

2. In the folder src/tests/params*files create a new_parameters file that has the name "constants*{name_of_your_test}" following the demo parameters file.

3. To lauch the training type in the source folder "python main_rnn_training.py "{name_of_your_test}".

The code will store the results of your test in the folder "tests_results".

## Testing the network

Many test functions are provided in the test package.
