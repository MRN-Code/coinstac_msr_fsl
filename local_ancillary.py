#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:00:16 2018

@author: Harshvardhan
"""
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def gather_local_stats(X, y, y_labels):

    biased_X = sm.add_constant(X)
    meanY_vector, lenY_vector = [], []

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []

    for column in y.columns:
        curr_y = list(y[column])
        meanY_vector.append(np.mean(curr_y))
        lenY_vector.append(len(y))

        # Printing local stats as well
        model = sm.OLS(curr_y, biased_X.astype(float)).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared_adj)

    keys = ["beta", "sse", "pval", "tval", "rsquared"]
    local_stats_list = []

    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_pvalues[index].tolist(), local_tvalues[index].tolist(),
            local_rsquared[index]
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)

    return meanY_vector, lenY_vector, local_stats_list
