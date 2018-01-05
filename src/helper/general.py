###########################
# helper functions
###########################

import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import yaml


class Convergence_Checker(object):
    def __init__(self, min_iters, max_iters, min_confirmations=1):
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.min_confirmations = min_confirmations
        self.reset()

    def reset(self):
        self.values = []

    def check(self, value):
        self.values.append(value)
        num_values = len(self.values)
        if num_values < self.min_iters:
            return False
        elif num_values >= self.max_iters:
            return True
        return (len(self.values) - np.argmin(self.values)
                >
                self.min_confirmations)

    def get_best(self):
        if len(self.values) > 0:
            return np.min(self.values)
        return -np.inf


def ensurePath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def askYN(question, default=-1):
    answers = '[y/n]'
    if default == 0:
        answers = '[N/y]'
    elif default == 1:
        answers = '[Y/n]'
    elif default != -1:
        raise Exception('Wrong default parameter (%d) to askYN!' % default)

    print(question + ' ' + answers)

    ans = input()

    if ans == 'y' or ans == 'Y':
        return True
    elif ans == 'n' or ans == 'N':
        return False
    elif len(ans) == 0:
        if default == 0:
            return False
        elif default == 1:
            return True
        elif default == -1:
            raise Exception('There is no default option given to this '
                            'y/n-question!')
        else:
            raise Exception('Logical error in askYN function!')
    else:
        raise Exception('Wrong answer to y/n-question! Answer was %s!' % ans)
    raise Exception('Logical error in askYN function!')


def create_batch_generator(batch_size, list_datasets):
    '''
    Important: all datasets in the list should have the same size, as they
    become randomly sampled preserving the correlation between the datasets,
    meaning that data entries in the same position of two sets keep occuring
    together in the same batch at the same position.
    '''
    if not isinstance(list_datasets, list):
        list_datasets = [list_datasets]
    nbr_datasets = len(list_datasets)
    n = len(list_datasets[0])
    for data in list_datasets:
        assert(n == len(data))
    perm = np.random.permutation(n)
    data_loaded = [data[perm] for data in list_datasets]
    nbr_epochs = 0
    while True:
        while len(data_loaded[0]) < batch_size:
            perm = np.random.permutation(n)
            for i in range(nbr_datasets):
                data_loaded[i] = np.concatenate((
                    data_loaded[i], list_datasets[i][perm]))
            nbr_epochs += 1
        result = []
        new_data_loaded = []
        for i in range(nbr_datasets):
            result_curr, data_loaded_curr = np.split(data_loaded[i],
                                                     [batch_size])
            result.append(result_curr)
            new_data_loaded.append(data_loaded_curr)
        data_loaded = new_data_loaded
        epoch_float = 1 - float(len(data_loaded[0])) / n
        assert(epoch_float >= 0.)
        assert(epoch_float <= 1.)

        yield tuple(result + [nbr_epochs + epoch_float])


NON_ALPHABETIC = re.compile('[^A-Za-z0-9_\-.=,:]')


def munge_filename(name):
    """Remove characters that might not be safe in a filename."""
    return NON_ALPHABETIC.sub('_', name)
