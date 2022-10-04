#! usr/bin/bash

#PATH='C:/Users/yes/Documents/Github/little_RNN/tests_results/'
#PATH='D:/Documents/NeuroRestore/Code/Github/little_RNN/tests_results/'

PATH=$(pwd)
PATH_PYTHON=/home/$(echo $USER)/anaconda3/envs/little_RNN/bin/python3

arg_list=( "[List of tests]")

echo $"Training of ${#arg_list[@]} networks..."
echo 
echo

cd $PATH

for i in ${arg_list[@]}
do 
    echo "Training " $i 
    $PATH_PYTHON main_rnn_training.py $i
done

echo "End of script"