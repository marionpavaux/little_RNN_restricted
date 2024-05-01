from typing import Tuple
import numpy as np
import pandas as pd
import torch
import sklearn.model_selection
from data import FS, PULSE_WIDTH


def augment_data(
    emg_array: np.ndarray,
    stim_arrays: np.ndarray,
    stim_features: pd.DataFrame,
    flexible_size: int = 50,
    n_new_data: int = 5,
    root: int = 2,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Applies an offset to the data that results in data augmentation
    :param emg_array: array on which to apply augmentation
    :param stim_features: dataframe to which will be added a delay column
    :param flexible size: maximum delay accorded in ms
    :param n_new_data: number of new data to generate
    :param offset: offset to apply to align the data in samples
    :return:
        augmented emg array and stim_features data frame
    """
    # convert flexible size and offset from ms to number of samples
    flexible_size = int(flexible_size * FS / 1000)

    augmented_emg_array = np.zeros(((n_new_data + 1) * emg_array.shape[0], emg_array.shape[1], emg_array.shape[2]))
    augmented_stim_arrays = np.zeros(
        ((n_new_data + 1) * stim_arrays.shape[0], stim_arrays.shape[1], stim_arrays.shape[2], stim_arrays.shape[3])
    )
    augmented_stim_features = (
        pd.concat([stim_features.reset_index(drop=True)] * (n_new_data + 1), axis=0).sort_index().reset_index(drop=True)
    )

    # choose a seed so that results does not depend on chance and randomly defined window_starts
    seed = np.random.seed(root)
    window_starts = np.random.randint(1, flexible_size, (emg_array.shape[0], n_new_data), dtype=np.int32)
    window_starts = np.insert(window_starts, 0, 1, axis=1)
    middle = int(
        (int(FS * (PULSE_WIDTH * 10 ** (-6))) + 1) // 2 + 20 * FS * 10 ** (-3)
    )  # add little something to not cut the stim at beginning for large variances

    for i_experiment, experiment in enumerate(window_starts):
        for i_start, start in enumerate(experiment):

            augmented_emg_array[(i_experiment * (n_new_data + 1) + i_start), :, :] = np.append(
                np.zeros((start, emg_array.shape[2])), emg_array[i_experiment, :-start, :], axis=0
            )
            augmented_stim_arrays[(i_experiment * (n_new_data + 1) + i_start), :, :, :] = np.append(
                np.zeros((stim_arrays.shape[1], start, stim_arrays.shape[3])),
                stim_arrays[i_experiment, :, :-start, :],
                axis=1,
            )

            augmented_emg_array[(i_experiment * (n_new_data + 1) + i_start), : middle + start, :] = 0

    augmented_stim_features["Delay"] = pd.Series(np.reshape(window_starts, len(augmented_stim_features), order="A"))
    return augmented_emg_array, augmented_stim_arrays, augmented_stim_features


def normalize(data_array: torch.Tensor, norm: np.ndarray = None, per_feature: bool = True) -> torch.Tensor:
    """
    Applies MaxAbsScaler normalization on tensor
    :param data_array: array on which to apply normalization
    :return:
        scaled_train: scaled training tensor
    """
    if per_feature:
        reducted_axis = tuple([i for i in range(len(data_array.shape) - 1)])
        MaxAbs = np.nanmax(np.abs(data_array), axis=reducted_axis) if norm is None else norm
        data_array = np.divide(data_array, MaxAbs, where=np.logical_and(MaxAbs != 0, np.invert(np.isnan(MaxAbs))))

    if not per_feature:
        MaxAbs = np.nanmax(np.abs(data_array)) if norm is None else norm
        data_array = np.divide(data_array, MaxAbs)

    data_array *= 100
    return data_array, MaxAbs


def train_test_split(
    input_features: pd.DataFrame, input_arrays: np.ndarray, label_array: np.ndarray, test_size: float = 0.2
) -> Tuple[np.ndarray]:
    """
    Splits data between train and test
    :param input_features: dataframe containing stim features
    :input_arrays: array of 4-dimensions containing input
    :label_array: array of 3-dimensions containing output
    :test_size: size of the training set
    :return:
        tuple containing trainig input features, testing input features, training input, testing input, training output, testing output
    """
    (
        train_input_features,
        test_input_features,
        train_input_arrays,
        test_input_arrays,
        train_label_array,
        test_label_array,
    ) = sklearn.model_selection.train_test_split(
        input_features, input_arrays, label_array, test_size=test_size, random_state=4
    )
    return (
        train_input_features,
        test_input_features,
        train_input_arrays,
        test_input_arrays,
        train_label_array,
        test_label_array,
    )
