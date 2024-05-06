# little RNN

This repository is the public/restricted version of the private repository little_RNN. I conducted this project at the Swiss laboratory .NeuroRestore in Lausanne in the Cyberspace team for my 6-month master's thesis. The project aims at creating an RNN that can reproduce the link between the time series of activation of 16 electrodes of the paddle lead .NeuroRestore implements in paraplegic patients, and the time series of activation of 14 muscles of the lower limb (EMGs). This model could be used for scientific understanding of the spinal circuits, as a simulation model for new data, or an optimization model while adding a controller.

The report and defense remain confidential because they expose medical data analysis.

## Approaches 

### Problematic 

Can we use a personalized mathematical model to understand and predict the muscle response caused by epidural stimulation for human lower limb muscles ? 

### Results 

#### Muscle activity as a function of the stimulation amplitude

| **Measured** | **Predicted** |
|----------|----------|
|<img src="https://github.com/marionpavaux/little_RNN_restricted/blob/main/results_plot_examples/stim_amplitude_vs_muscle_response/measured/cath2_an3_8.png" title="measured"  alt="measured" height="400"/> | <img src="https://github.com/marionpavaux/little_RNN_restricted/blob/main/results_plot_examples/stim_amplitude_vs_muscle_response/predicted/cath2_an3_8.png" title="predicted" alt="predicted" height="400"/> |

### Selectivity index as a function of the stimulation amplitude 

| **Measured** | **Predicted** |
|----------|----------|
|<img src="https://github.com/marionpavaux/little_RNN_restricted/blob/main/results_plot_examples/stim_vs_selectivity_index/measured/cath2_an3_8.png" title="measured"  alt="measured" height="400"/> | <img src="https://github.com/marionpavaux/little_RNN_restricted/blob/main/results_plot_examples/stim_vs_selectivity_index/predicted/cath2_an3_8.png" title="predicted" alt="predicted" height="400"/> |

### Network periodicity 
| **20 Hz stimulation** | **40 Hz stimulation** |
|----------|----------|
|<img src="https://github.com/marionpavaux/little_RNN_restricted/blob/main/results_plot_examples/pca_20Hz.png" title="20 Hz"  alt="20 Hz" height="400"/> | <img src="https://github.com/marionpavaux/little_RNN_restricted/blob/main/results_plot_examples/pca_40Hz.png" title="40 Hz" alt="40 Hz" height="400"/> |


## Repository

In the folder plot_examples, you can take a look at some resulting plots of the `main_rnn_training.py` script. There was many more but I only chose those that do not compromise medical data.

The source folder (src) is composed of three packages, one intended for data processing and analysis (data), one for the training of the rnn (rnn) and finally one for the training of the controller (controller).

### Workspace setup

1. create conda environment:
   `bash
conda create -f environment.yml
`

2. install specific pytorch tools if you have a GPU (see https://pytorch.org/get-started/locally/)

### Data

Data description and format is restricted.

### RNN training

To train a RNN :

1. Check the constant parameters in the **init**.py file of each package

2. In the folder src/tests/params*files create a new_parameters file that has the name "constants*{name_of_your_test}" following the demo parameters file.

3. To lauch the training type in the source folder `python main_rnn_training.py "{name_of_your_test}"`.

The code will store the results of your test in the folder "tests_results".

## Testing the network

Many test functions are provided in the test package.
