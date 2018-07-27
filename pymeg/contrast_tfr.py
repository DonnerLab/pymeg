import os
import pandas as pd
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from joblib import Memory

from pymeg import atlas_glasser

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)


def load_tfr_data(globstring):  # subj, session, timelock, data_folder):
    """Load all files identified by glob string"""
    tfr_data_filenames = glob(globstring)
    tfrs = []
    for f in tfr_data_filenames:
        tfr = pd.read_hdf(f)
        tfr = pd.pivot_table(tfr.reset_index(), values=tfr.columns, index=[
                             'trial', 'est_val'], columns='time').stack(-2)
        tfr.index.names = ['trial', 'freq', 'area']
        tfrs.append(tfr)
    tfr = pd.concat(tfrs)
    return tfr


def baseline_per_sensor_get(tfr, baseline_time=(-0.25, -0.15)):
    '''
    Get average baseline
    '''
    time = tfr.columns.get_level_values('time').values.astype(float)
    id_base = (time > baseline_time[0]) & (time < baseline_time[1])
    base = tfr.loc[:, id_base].groupby(['freq', 'area']).mean().mean(
        axis=1)  # This should be len(nr_freqs * nr_hannels)
    return base


def baseline_per_sensor_apply(tfr, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    def div(x):
        freqs = x.index.get_level_values('freq').values[0]
        areas = x.index.get_level_values('area').values[0]
        bval = float(baseline
                     .loc[
                         baseline.index.isin([freqs], level='freq')
                         & baseline.index.isin([areas], level='area')])
        return (x - bval) / bval * 100
    return tfr.groupby(['freq', 'area']).apply(div)


@memory.cache
def load_tfr_contrast(data_globstring, base_globstring, meta_data, conditions,
                      baseline_time, n_jobs=4):
    """Load a set of data files and turn them into contrasts.
    """

    tfrs = []
    # load data:
    tfr_data = load_tfr_data(data_globstring)
    # Make sure that meta_data and tfr_data overlap in trials
    tfr_trials = np.unique(tfr_data.index.get_level_values('trial').values)
    meta_trials = np.unique(meta_data.reset_index().loc[:, 'hash'].values)
    assert(any([t in meta_trials for t in tfr_trials]))

    # data to baseline:
    if not (data_globstring == base_globstring):
        tfr_data_to_baseline = load_tfr_data(base_globstring)
    else:
        tfr_data_to_baseline = tfr_data

    areas = np.unique(tfr_data.index.get_level_values('area'))

    # compute contrasts:
    from itertools import product
    tasks = []
    for area, condition in product(areas, conditions):
        tasks.append((tfr_data, tfr_data_to_baseline, meta_data,
                      area, condition, baseline_time))

    tfr_conditions = Parallel(n_jobs=n_jobs, verbose=1, backend='threading')(
        delayed(make_tfr_contrasts)(*task) for task in tasks)

    tfrs.append(pd.concat(tfr_conditions))
    tfrs = pd.concat(tfrs)
    return tfrs


def make_tfr_contrasts(tfr_data, tfr_data_to_baseline, meta_data, area,
                       condition, baseline_time):

    # unpack:
    condition_ind = meta_data.loc[meta_data[condition] == 1, "hash"]

    # apply condition ind, collapse across trials, and get baseline::
    tfr_data_to_baseline = (tfr_data_to_baseline
                            .loc[
                                tfr_data_to_baseline.index.isin(condition_ind, level='trial') &
                                tfr_data_to_baseline.index.isin([area], level='area'), :]
                            .groupby(['freq', 'area']).mean())

    baseline = baseline_per_sensor_get(
        tfr_data_to_baseline, baseline_time=baseline_time)

    # apply condition ind, and collapse across trials:
    tfr_data_condition = (tfr_data
                          .loc[
                              tfr_data.index.isin(condition_ind, level='trial') &
                              tfr_data.index.isin([area], level='area'), :]
                          .groupby(['freq', 'area']).mean())

    # apply baseline, and collapse across sensors:
    tfr_data_condition = baseline_per_sensor_apply(
        tfr_data_condition, baseline=baseline).groupby(['freq', ]).mean()

    tfr_data_condition['area'] = area
    tfr_data_condition['condition'] = condition
    tfr_data_condition = tfr_data_condition.set_index(
        ['area', 'condition', ], append=True, inplace=False)
    tfr_data_condition = tfr_data_condition.reorder_levels(
        ['area', 'condition', 'freq'])
    #tfr_data_condition = tfr_data_condition.set_index(
    #    ['area'], append=True, inplace=False)
    #tfr_data_condition = tfr_data_condition.reorder_levels(
    #    ['area', 'freq'])
    return tfr_data_condition


@memory.cache
def compute_contrast(contrast, weights, hemi, data_globstring, base_globstring,
                     meta_data, baseline_time, n_jobs=15):
    """Compute a single contrast from tfr data
    Args:
        contrast: list
            A list of columns in meta_data that are 1 whenever a trial is part
            of a condition to be contrasted
        weights: list
            A list of weights for each condition that determines the contrast
            to be computed
        hemi: str
            Can be:
                'lh_is_ipsi' if contrast is lateralized and lh is ipsi
                'rh_is_ipsi' if contrast is lateralized and rh is ipsi
                'avg' if contrast should be averaged across hemispheres
        data_globstring: string
            A string that selects a set of filenames if passed through
            glob.
        base_globstring: string
            Same as data_globstring but selects data to use for baselining
        meta_data: data frame
            Meta data DataFrame with as many rows as trials.
        baseline_time: tuple

    """
    all_clusters, visual_field_clusters, glasser_clusters, jwg_clusters = atlas_glasser.get_clusters()
    print('redo')
    # load for all subjects:
    tfr_condition = []

    tfr_condition.append(
        load_tfr_contrast(data_globstring, base_globstring, meta_data,
                          contrast, baseline_time, n_jobs=n_jobs))
    tfr_condition = pd.concat(tfr_condition)

    # mean across sessions:
    tfr_condition = tfr_condition.groupby(
        ['area', 'condition', 'freq']).mean()
    cluster_contrasts = []
    import functools
    for cluster in all_clusters.keys():
        right = []
        left = []
        for condition in contrast:
            tfrs_rh = []
            tfrs_lh = []
            for area in all_clusters[cluster]:
                area_idx = tfr_condition.index.isin([area], level='area')
                condition_idx = tfr_condition.index.isin(
                    [condition], level='condition')
                subset = tfr_condition.loc[area_idx & condition_idx].groupby(                    
                    ['freq']).mean()
                if 'rh' in area:
                    tfrs_rh.append(subset)
                else:
                    tfrs_lh.append(subset)

            left.append(pd.concat(tfrs_lh))
            right.append(pd.concat(tfrs_rh))
        if hemi == 'lh_is_ipsi':
            tfrs = [left[i] - right[i]
                    for i in range(len(left))]
        elif hemi == 'rh_is_ipsi':
            tfrs = [right[i] - left[i]
                    for i in range(len(left))]
        else:
            tfrs = [(right[i] + left[i]) / 2
                    for i in range(len(left))]
        assert(len(tfrs) == len(weights))
        tfrs = [tfr * weight for tfr, weight in zip(tfrs, weights)]
        tfrs = functools.reduce(lambda x, y: x + y, tfrs)
        tfrs.loc[:, 'cluster'] = cluster
        cluster_contrasts.append(tfrs)
    return pd.concat(cluster_contrasts)


def augment_data(meta, response_left, stimulus):
    """Augment meta data with fields for specific cases

    Args:
        meta: DataFrame
        response_left: ndarray
            1 if subject made a left_response / yes response
        stimulus: ndarray
            1 if a left_response is correct
    """
    # add columns:
    meta["all"] = 1

    meta["left"] = response_left.astype(int)
    meta["right"] = (~response_left).astype(int)

    meta["hit"] = ((response_left == 1) & (stimulus == 1)).astype(int)
    meta["fa"] = ((response_left == 1) & (stimulus == 0)).astype(int)
    meta["miss"] = ((response_left == 0) & (stimulus == 1)).astype(int)
    meta["cr"] = ((response_left == 0) & (stimulus == 0)).astype(int)
    return meta
