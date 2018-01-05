#!/usr/bin/env python

'''
helper/plotting.py: A module providing simple plotting routines.
'''
###############################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def create_histogram(list_datasets, names_datasets, variable, out_path):
    matplotlib.rcParams.update({'font.size': 13})
    if not isinstance(list_datasets, list):
        list_datasets = [list_datasets]
    if not isinstance(names_datasets, list):
        names_datasets = [names_datasets]
    all_data = list_datasets[0]
    for dataset in list_datasets[1:]:
        all_data = np.concatenate([all_data, dataset])
    _, bin_edges = np.histogram(all_data, bins='auto')
    plt.figure()
    plt.grid()
    y_min = np.inf
    for dataset, name in zip(list_datasets, names_datasets):
        weights = np.ones(dataset.shape) / dataset.shape[0]
        counts, _ = np.histogram(dataset, bins=bin_edges)
        y, bin_edges, p = plt.hist(dataset, histtype='step', bins=bin_edges,
                                   weights=weights, lw=2, label=name)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.errorbar(
            bin_centers,
            y,
            yerr=counts**0.5 / dataset.shape[0],
            marker='.',
            drawstyle='steps-mid',
            capsize=2,
            alpha=0.6,
            fmt='none',
            color=p[0].get_edgecolor()
        )
        y_min = min(y_min, np.min(y[y != 0.0]))
    plt.xlabel(variable)
    plt.ylabel('fraction of data')
    plt.legend()
    plt.yscale('symlog', linthreshy=y_min, linscaley=0.1)
    plt.savefig(out_path)
    plt.close()
