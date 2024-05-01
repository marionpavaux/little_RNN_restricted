import numpy as np
from data import MUSCLES


def get_muscle_response(emg_array: np.ndarray, reducted_axis: tuple) -> np.ndarray:
    """
    Gets the muscle reponse for each muscle in the emg_array.

    :param emg_array: emg array
    :param reducted_axis: tuple of axis to reduce
    :return:
        maximum absolute muscle response, taken along reduced_axis.
    """
    MaxAbs = np.nanmax(np.abs(emg_array), axis=reducted_axis)
    return MaxAbs


def get_SI(emg_array: np.ndarray, muscle: str, reducted_axis: int = 0) -> np.ndarray:
    """
    Gets the selectivity index for given muscle and each experiment in the array.

    :param emg_array: emg array
    :param muscle: muscle to calculate selectivity index
    :param reducted_axis: axis to reduce
    :return:
        selectivity index of the muscle for each experiment.
    """
    muscle_ind = MUSCLES.index(muscle)
    MaxAbs = get_muscle_response(emg_array, reducted_axis)
    SI_muscle = MaxAbs[:, muscle_ind] / (1 + np.sum(MaxAbs, axis=1, where=np.asarray(muscle != np.array(MUSCLES))))
    return SI_muscle
