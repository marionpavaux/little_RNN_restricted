# little RNN

This repository is the public/restricted version of the private repository little_RNN. I conducted this project at the Swiss laboratory .NeuroRestore in Lausanne in the Cyberspace team for my 6-months master's thesis. The project aims at creating a RNN, trained by backpropagation, that can reproduce the link between the time series of activation of 16 electrodes of the paddle lead .NeuroRestore implements in paraplegic patients, and the time series of activation of 14 muscles of the lower limb (emgs). This model could be use for scientific understanding of the spinal circuits, as a simulation model for new data, or an optimization model while adding a controller.

The report and defense remain confidential because they expose medical data analysis.

In the folder plot_examples, you can take a look at some resulting plots of the main_rnn_training.py script. There was many more but I only chose those that do not compromise medical data. 

The source folder (src) is composed of three packages, one intended for data processing and analysis (data), one for the training of the rnn (rnn) and finally one for the training of the controller (controller).  

## Environment creation ##

<ol>

<li>create conda environment with name little_RNN</li>
conda create -n little_RNN anaconda python=3.9

<li>activate environment 
conda activate little_RNN

<li>install pytorch</li> 
follow instruction on pytorch webpage get started : https://pytorch.org/get-started/locally/ 

<li>update the environment 
conda update --all 

</ol>

## Data ## 

Next to the src folder create a folder named "tests_results". Put your data folder in "tests_results".

Data description and format is restricted. 

## RNN training ## 

To train a RNN : 

<ol>

<li> Check the constant parameters in the __init__.py file of each package</li>

<li> In the folder src/tests/params_files create a new_parameters file that has the name "constants_{name_of_your_test}" following the demo parameters file.</li>

<li> To lauch the training type in the source folder "python main_rnn_training.py {name_of_your_test}".</li>

The code will store the results of your test in the folder "tests_results".

</ol>

## Testing the network ##

Many test functions are provided in the test package. 
