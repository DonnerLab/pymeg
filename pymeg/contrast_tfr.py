import os
import pandas as pd
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from joblib import Memory
import logging

from pymeg import atlas_glasser

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)

backend = 'loky'


class Cache(object):
    """A cache that can prevent reloading from disk.

    Can be used as a context manager.
    """

    def __init__(self, cache=True):
        self.store = {}
        self.cache = cache

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.clear()

    def get(self, globstring):
        if self.cache:
            if globstring not in self.store:
                self.store[globstring] = self._load_tfr_data(globstring)
            else:
                logging.info('Returning cached object: %s' % globstring)
            return self.store[globstring]
        else:
            return self._load_tfr_data(globstring)

    def clear(self):
        self.cache = {}

    def _load_tfr_data(self, globstring):
        """Load all files identified by glob string"""
        logging.info('Loading data for: %s' % globstring)
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


@memory.cache(ignore=['cache'])
def load_tfr_contrast(data_globstring, base_globstring, meta_data, conditions,
                      baseline_time, n_jobs=1, cache=Cache(cache=False)):
    """Load a set of data files and turn them into contrasts.
    """
    tfrs = []
    # load data:
    tfr_data = cache.get(data_globstring)
    # Make sure that meta_data and tfr_data overlap in trials
    tfr_trials = np.unique(tfr_data.index.get_level_values('trial').values)
    meta_trials = np.unique(meta_data.reset_index().loc[:, 'hash'].values)
    assert(any([t in meta_trials for t in tfr_trials]))

    # data to baseline:
    if not (data_globstring == base_globstring):
        tfr_data_to_baseline = cache.get(base_globstring)
    else:
        tfr_data_to_baseline = tfr_data

    areas = np.unique(tfr_data.index.get_level_values('area'))

    # compute contrasts:
    from itertools import product
    tasks = []
    for area, condition in product(areas, conditions):
        tasks.append((tfr_data, tfr_data_to_baseline, meta_data,
                      area, condition, baseline_time))

    tfr_conditions = Parallel(n_jobs=n_jobs, verbose=1, backend=backend)(
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
    return tfr_data_condition


#@memory.cache(ignore=['cache'])
def compute_contrast(contrast, weights, hemi, data_globstring, base_globstring,
                     meta_data, baseline_time, n_jobs=15, cache=Cache(cache=False)):
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
    # load for all subjects:
    tfr_condition = []

    tfr_condition.append(
        load_tfr_contrast(data_globstring, base_globstring, meta_data,
                          contrast, baseline_time, n_jobs=n_jobs, cache=cache))
    tfr_condition = pd.concat(tfr_condition)

    # mean across sessions:
    tfr_condition = tfr_condition.groupby(
        ['area', 'condition', 'freq']).mean()
    cluster_contrasts = []
    import functools
    logging.info('Start computing contrast %s for clusters' % contrast)
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
        tfrs = tfrs.groupby('freq').mean()
        tfrs.loc[:, 'cluster'] = cluster
        cluster_contrasts.append(tfrs)
    logging.info('Done compute contrast')
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


def plot_contrasts(tfr_data, contrasts, areas):
    all_clusters, vf_clusters, glasser_clusters, jwg_clusters = atlas_glasser.get_clusters()
    areas = ['vfcPrimary', 'vfcEarly', 'vfcVO', 'vfcPHC', 'vfcV3ab',
             'vfcTO', 'vfcLO', 'vfcIPS01', 'vfcIPS23', 'vfcFEF',
             'JWG_aIPS', 'JWG_IPS_PCeS', 'JWG_M1']
    areas += glasser_clusters.keys()

    for row, area in enumerate(areas):
        for col, contrast in enumerate(contrasts):
            plt.subplot(len(areas), len(contrasts * 2))
            data = tfr_data.query(
                'area=="%s" & contrast=="%s" % epoch=="stimulus"')
            plot_tfr(tfr, (-0.25, 1.35), -15, 15, 'stimulus')


def set_jw_style():
    import matplotlib
    import seaborn as sns
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    sns.set(style='ticks', font='Arial', font_scale=1, rc={
        'axes.linewidth': 0.25,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'xtick.major.width': 0.25,
        'ytick.major.width': 0.25,
        'text.color': 'Black',
        'axes.labelcolor': 'Black',
        'xtick.color': 'Black',
        'ytick.color': 'Black', })
    sns.plotting_context()


def plot_tfr(tfr, time_cutoff, vmin, vmax, tl, cluster_correct=False, threshold=0.05, ax=None):
    from mne.stats import permutation_cluster_1samp_test as permutation_test
    import pylab as plt
    # colorbar:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)

    # variables:
    times = np.array(tfr.columns, dtype=float)
    freqs = np.array(
        np.unique(tfr.index.get_level_values('freq')), dtype=float)
    time_ind = (times > time_cutoff[0]) & (times < time_cutoff[1])
    time_ind = (times > time_cutoff[0]) & (times < time_cutoff[1])

    # data:
    X = np.stack(
        [tfr.loc[tfr.index.isin([subj], level='subj'), time_ind].values
         for subj in np.unique(tfr.index.get_level_values('subj'))]
    )

    # grand average plot:
    cax = ax.pcolormesh(times[time_ind], freqs, X.mean(
        axis=0), vmin=vmin, vmax=vmax, cmap=cmap)

    # cluster stats:
    if cluster_correct:
        if tl == 'stimulus':
            test_data = X[:, :, times[time_ind] > 0]
            times_test_data = times[time_ind][times[time_ind] > 0]
        else:
            test_data = X.copy()
            times_test_data = times[time_ind]
        try:
            T_obs, clusters, cluster_p_values, h0 = permutation_test(
                test_data, threshold={'start': 0, 'step': 0.2},
                connectivity=None, tail=0, n_permutations=1000, n_jobs=10)
            sig = cluster_p_values.reshape(
                (test_data.shape[1], test_data.shape[2]))
            ax.contour(times_test_data, freqs, sig, (threshold,),
                       linewidths=0.5, colors=('black'))
        except:
            pass

    ax.axvline(0, ls='--', lw=0.75, color='black',)

    if tl == 'stimulus':
        ax.set_xlabel('Time from stimulus (s)')
        ax.axvline(-0.25, ls=':', lw=0.75, color='black',)
        ax.axvline(-0.15, ls=':', lw=0.75, color='black',)
        ax.set_ylabel('Frequency (Hz)')
        # ax.set_title('{} contrast'.format(c))
    else:
        ax.set_xlabel('Time from report (s)')
        # ax.set_title('N = {}'.format(len(subjects)))
        ax.tick_params(labelleft='off')
        plt.colorbar(cax, ticks=[vmin, 0, vmax])

    return ax
