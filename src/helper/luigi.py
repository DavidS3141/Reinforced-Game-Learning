#!/usr/bin/env python

'''
helper/luigi.py: A module providing simple luigi helpers.
'''
###############################################################################


def safe_parameter_passing(task, **kwargs):
    common_params = list(set.intersection(set(kwargs.keys()),
                                          set(task.get_param_names(
                                              include_significant=True))))
    common_kwargs = dict([(key, kwargs[key]) for key in common_params])
    return task(**common_kwargs)
