from __future__ import division
from __future__ import print_function
'''
Compute TFRs from source reconstructed data

--- Important: Not yet independent form Niklas's experiment ---

This module works on average TFRs for specific ROIS. It takes the output from
source reconstruction and reduces this to average TFRs.

The logical path through this module is:

1) load_sub_grouped reduces RAW data to average tfrs. It's a good idea
   to call this function for each subject on the cluster to distribute
   memory load. Output will be cached.
2) To compute a contrast call load_sub_grouped with a filter_dict. This
   allows to compute average TFRs for sub groups of trials per subject.
3) Use the plotting functions to plot TFRs for different ROIs.
'''


import os
if 'DISPLAY' in os.environ.keys():
    try:
        from surfer import Brain
    except:
        Brain = None
        print('No pysurfer support')

import numpy as np
import pandas as pd

from conf_analysis.meg import preprocessing
from pymeg import roi_clusters as rois

from joblib import Memory

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'])

'''
The following functions will reduce raw data to various different average TFRs.
'''


def baseline(data, baseline_data, baseline=(-0.25, 0)):
    baseline_data = baseline_data.query('%f < time & time < %f' % baseline)
    m = baseline_data.mean()
    s = baseline_data.std()
    # print(m, s)
    return (data.subtract(m, 1)
                .div(s, 1))


def contrast_controlled_response_contrast(sub, epoch='stimulus'):
    '''
    Compute a response contrast but control for
    mean contrast level. To this by splitting up
    trials into several contrast groups and then computing
    individual response contrasts within.
    '''
    meta = preprocessing.get_meta_for_subject(sub, 'stimulus')
    for (response, mc) in meta.groupby(['response']):
        pass,
    filter_dict = {'M1': meta.query('response==-1').reset_index().loc[:, 'hash'].values,
                   'P1': meta.query('response==1').reset_index().loc[:, 'hash'].values}
    if sub <= 8:
        hand_mapping = {'M1': 'lh_is_ipsi', 'P1': 'rh_is_ipsi'}
    else:
        hand_mapping = {'M1': 'rh_is_ipsi', 'P1': 'lh_is_ipsi'}
    return contrast(sub, filter_dict, hand_mapping, ['P1', 'M1'],
                    epoch=epoch, baseline_time=baseline_time)


def response_contrast(subs=range(1, 16), epoch='stimulus'):
    return pd.concat(
        [_prewarm_response_contrast(sub, epoch=epoch) for sub in subs])


def _prewarm_response_contrast(sub, epoch='stimulus', baseline_time=(-0.25, 0)):
    meta = preprocessing.get_meta_for_subject(sub, 'stimulus')
    filter_dict = {'M1': meta.query('response==-1').reset_index().loc[:, 'hash'].values,
                   'P1': meta.query('response==1').reset_index().loc[:, 'hash'].values}
    if sub <= 8:
        hand_mapping = {'M1': 'lh_is_ipsi', 'P1': 'rh_is_ipsi'}
    else:
        hand_mapping = {'M1': 'rh_is_ipsi', 'P1': 'lh_is_ipsi'}
    return contrast(sub, filter_dict, hand_mapping, ['P1', 'M1'],
                    epoch=epoch, baseline_time=baseline_time)

#@memory.cache
def contrast(sub, filter_dict, hand_mapping, contrast,
             epoch='stimulus', baseline_epoch='stimulus',
             baseline_time=(-0.25, 0)):
    tfrs = []

    trial_list = filter_dict.values()
    condition_names = filter_dict.keys()
    condition_tfrs, weights = load_sub_grouped_weighted(
        sub, trial_list, epoch=epoch)

    if baseline_epoch == epoch:
        baseline_tfrs = [t.copy() for t in condition_tfrs]
    else:
        baseline_tfrs, weights = load_sub_grouped_weighted(
            sub, trial_list, epoch=baseline_epoch)

    for condition, tfr, base_tfr in zip(condition_names, condition_tfrs, baseline_tfrs):
        tfr.loc[:, 'condition'] = condition
        # tfr.set_index('condition', append=True, inplace=True)
        base_tfr.loc[:, 'condition'] = condition
        # base_tfr.set_index('condition', append=True, inplace=True)
        # Baseline correct here
        tfr = tfr.reset_index().set_index(
            ['sub', 'est_key', 'est_val', 'condition', 'time'])
        base_tfr.set_index('condition', append=True, inplace=True)
        groups = []
        base_tfr = base_tfr.reset_index().set_index(
            ['sub', 'est_key', 'est_val', 'condition', 'time'])
        for gp, group in tfr.groupby(['sub', 'est_key',
                                      'est_val', 'condition']):
            base = base_tfr.loc[gp, :]
            group = baseline(group, base, baseline=baseline_time)
            groups.append(group)

        tfr = pd.concat(groups)

        # Now compute lateralisation
        left, right = rois.lh(tfr.columns), rois.rh(tfr.columns)
        if hand_mapping[condition] == 'lh_is_ipsi':
            print(sub, condition, 'Left=IPSI')
            ipsi, contra = left, right
        elif hand_mapping[condition] == 'rh_is_ipsi':
            print(sub, condition, 'Right=IPSI')
            ipsi, contra = right, left
        else:
            raise RuntimeError('Do not understand hand mapping')
        lateralized = rois.lateralize(tfr, ipsi, contra)

        # Averge hemispheres
        havg = pd.concat(
            [tfr.loc[:, (x, y)].mean(1) for x, y in zip(left, right)],
            1)
        havg.columns = [x + '_Havg' for x in left]

        tfrs.append(pd.concat([tfr, lateralized, havg], 1))
    tfrs = pd.concat(tfrs, 0)
    #cond1 = tfrs.query('condition=="%s"' % contrast[0])
    #cond2 = tfrs.query('condition=="%s"' % contrast[1])
    # delta = (cond1.reset_index(level='condition', drop=True) -
    #         cond2.reset_index(level='condition', drop=True))
    #delta.loc[:, 'condition'] = 'diff'
    #delta.set_index('condition', append=True, inplace=True)
    return tfrs  # pd.concat([tfrs, delta], )


def load_sub_grouped_weighted(sub, trials, epoch='stimulus'):
    '''
    Load average TFR values generated from a specific
    set of trials.

    Trials from different sessions are loaded and averaged per
    session. Session averages are computed by weighting each
    session with its fraction of trials.

    Parameters:
        sub : int, subject number
        trials : list of trial hashes
    '''
    sacc = []
    stdacc = []
    weights = []
    for session in range(4):
        try:
            df, stds, ws = load_sub_session_grouped(
                sub, session, trials, epoch=epoch)
            sacc.append(df)
            stdacc.append(stds)
            weights.append(ws)
        except IOError:
            print('No data for %i ,%i' % (sub, session))
    n_conditions = len(sacc[0])
    conditions = []
    stds = []

    for cond in range(n_conditions):
        averages = pd.concat([s[cond] for s in sacc]).groupby(
            ['sub', 'time', 'est_key', 'est_val']).mean()
        conditions.append(averages)

    return conditions, weights


@memory.cache
def load_sub_session_grouped(sub, session, trial_list, epoch='stimulus',
                             baseline=(-0.25, 0), log10=True):

    if epoch == 'stimulus':
        df = pd.read_hdf(
            '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-stimulus-lcmv.hdf' % (
                sub, session))
    elif epoch == 'response':
        df = pd.read_hdf(
            '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-response-lcmv.hdf' % (
                sub, session))
    else:
        raise RuntimeError('Do not understand epoch %s' % epoch)
    df.set_index(['trial', 'time',
                  'est_key', 'est_val'], inplace=True)
    if log10:
        df = np.log10(df) * 10

    df = rois.reduce(df).reset_index()  # Reduce to visual clusters
    all_trials = df.loc[:, 'trial']
    conditions = []
    weights = []
    stds = []
    for trials in trial_list:
        id_trials = np.in1d(all_trials, trials)
        n_trials = len(np.unique(df.loc[id_trials, 'trial'].values))
        df_sub = df.set_index('trial').loc[trials, :].reset_index()
        weights.append(float(n_trials) / len(trials))
        df_sub.loc[:, 'sub'] = sub
        df_sub.loc[:, 'session'] = session

        conditions.append(df_sub.groupby(
            ['session', 'sub', 'time', 'est_key', 'est_val']).mean())

        stds.append((
            df_sub.query('%f < time & time < %f' % baseline)
                  .groupby(['session', 'sub', 'est_key', 'est_val'])
                  .std()))
    return conditions, stds, weights


@memory.cache
def stats_test(data, n_permutations=1000):
    from mne.stats import permutation_cluster_1samp_test
    threshold_tfce = dict(start=0, step=0.2)
    t_tfce, _, p_tfce, H0 = permutation_cluster_1samp_test(
        data.swapaxes(1, 2), n_jobs=4, threshold=threshold_tfce, connectivity=None, tail=0,
        n_permutations=n_permutations)
    return t_tfce.T, p_tfce.reshape(t_tfce.shape).T, H0


def get_tfr_stack(data, area, baseline=None, tslice=slice(-0.25, 1.35)):
    stack = []
    for sub, ds in data.groupby('sub'):
        stack.append(get_tfr(ds, area, tslice=tslice).values)
    return np.stack(stack)


def get_tfr(data, area, baseline=None, tslice=slice(-0.25, 1.35)):
    k = pd.pivot_table(data.reset_index(), values=area,
                       index='est_val', columns='time')
    if baseline is not None:
        k = k.subtract(k.loc[:, baseline].mean(1), axis=0).div(
            k.loc[:, baseline].std(1), axis=0)
    return k.loc[:, tslice]

'''
The following functions allow plotting of TFRs.
'''


def plot_set(response, stimulus, setname, setareas, minmax=(10, 20),
             lateralize=False, stats=False,
             response_tslice=slice(-0.5, 0.5),
             stimulus_tslice=slice(-0.25, 1.35),
             new_figure=False):
    from matplotlib.gridspec import GridSpec
    import pylab as plt

    columns = response.columns
    if lateralize:
        columns = rois.filter_cols(columns, ['Lateralized'])
    else:
        columns = rois.filter_cols(columns, ['Havg'])
    areas = rois.filter_cols(columns, setareas)
    # Setup gridspec to compare stimulus and response next to each other.
    rows, cols = rois.layouts[setname]
    if new_figure:
        plt.figure(figsize=(cols * 3.5, rows * 3.5))

    gs = GridSpec(2 * rows, 2 * cols, height_ratios=[140, 16] * rows,
                  width_ratios=[1.55, 1] * cols)
    locations = []

    # First plot stimulus and comput stimulus positions in plot.
    for ii, area in enumerate(setareas):
        row, col = (ii // cols) * 2, np.mod(ii, cols)
        locations.append((row, col * 2))
    plot_labels(stimulus, areas, locations, gs,
                minmax=minmax, stats=stats, tslice=stimulus_tslice)

    # First plot stimulus and comput stimulus positions in plot.
    locations = [(row, col + 1) for row, col in locations]

    plot_labels(response, areas, locations, gs,
                minmax=minmax, stats=stats, tslice=response_tslice)


def plot_labels(data, areas, locations, gs, stats=True, minmax=(10, 20),
                tslice=slice(-0.25, 1.35)):
    '''
    Plot TFRS for a set of ROIs. At most 6 labels.
    '''
    labels = rois.filter_cols(data.columns, areas)
    import pylab as plt
    #import seaborn as sns
    #colors = sns.color_palette('bright', len(labels))

    p = None
    maxrow = max([row for row, col in locations])
    maxcol = max([row for row, col in locations])

    for (row, col), area in zip(locations, labels):

        plt.subplot(gs[row, col])
        ex_tfr = get_tfr(data.query('est_key=="F"'), area, tslice=tslice)
        s = get_tfr_stack(data.query('est_key=="F"'), area, tslice=tslice)
        if stats:
            t, p, H0 = stats_test(s)
            p = p.reshape(t.shape)
        cbar = _plot_tfr(area, ex_tfr.columns.values, ex_tfr.index.values,
                         s.mean(0), p, title_color='k', minmax=minmax[0])
        if ((row + 2, col + 1) == gs.get_geometry()):
            pass
        else:
            cbar.remove()
        plt.xticks([])

        if col > 0:
            plt.yticks([])
        else:
            plt.ylabel('Freq')
        plt.subplot(gs[row + 1, col])
        ex_tfr = get_tfr(data.query('est_key=="LF"'), area, tslice=tslice)
        s = get_tfr_stack(data.query('est_key=="LF"'), area, tslice=tslice)
        if stats:
            t, p, H0 = stats_test(s)
            p = p.reshape(t.shape)
        cbar = _plot_tfr(area, ex_tfr.columns.values, ex_tfr.index.values,
                         s.mean(0), p, title_color='k', minmax=minmax[1])
        cbar.remove()
        #plt.xticks([0, 0.5, 1])
        if row == maxrow:
            plt.xlabel('time')

            #plt.xticks([tslice.start, 0, tslice.stop])
        else:
            plt.xticks([])


def plot_tfr(data, area, ps=None, minmax=None, title_color='k'):
    tfr = get_tfr(data, area)
    tfr_values = get_tfr_stack(data, area).mean(0)
    _plot_tfr(area, tfr.columns.values, tfr.index.values,
              tfr_values, ps, title_color=title_color, minmax=minmax)


def _plot_tfr(area, columns, index, tfr_values, ps, title_color='k',
              minmax=None):
    import pylab as plt
    import seaborn as sns

    if minmax is None:
        minmax = np.abs(np.percentile(tfr_values, [1, 99])).max()
    di = np.diff(index)[0] / 2.
    plt.imshow(np.flipud(tfr_values), cmap='RdBu_r', vmin=-minmax, vmax=minmax,
               extent=[columns[0], columns[-1],
                       index[0] - di, index[-1] + di], aspect='auto',
               interpolation='none')

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    sns.despine(ax=plt.gca(), bottom=True)
    cbar = plt.colorbar()
    cbar.set_ticks([np.ceil(-minmax), np.floor(minmax)])
    if ps is not None:
        plt.contour(ps, [0.05], extent=[columns[0], columns[-1], index[0] - di, index[-1] + di],
                    origin='lower')

    tarea = (area
             .replace('lh.JWDG.', '')
             .replace('lh.a2009s.', '')
             .replace('rh.JWDG.', '')
             .replace('rh.a2009s.', '')
             .replace('lh.wang2015atlas.', '')
             .replace('rh.wang2015atlas.', '')
             .replace('-lh_Lateralized', '')
             .replace('-lh_Havg', ''))

    plt.xlim(columns[0], columns[-1])
    plt.xticks([0])
    if index.max() > 30:
        plt.title(tarea, color=title_color)
        plt.yticks([10,  50, 100,  150])
        plt.ylim([10 - di, 150 + di])
    else:
        plt.yticks([4, 20])
        plt.ylim([4 - di, 20 + di])
    return cbar


'''
Deprecated
'''


@memory.cache
def load_sub_grouped(sub, trials=None, epoch='stimulus'):
    '''
    Filter trials should either be none or a dict with keys describing
    conditions and values a list of trials belonging to this condition.

    Parameters:
        sub : int, subject number
        trials : list of trial hashes
    '''
    sacc = []
    for session in range(4):
        if epoch == 'stimulus':
            df = pd.read_hdf(
                '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-stimulus-lcmv.hdf' % (
                    sub, session))
        elif epoch == 'respone':
            df = pd.read_hdf(
                '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-response-lcmv.hdf' % (
                    sub, session))
        else:
            raise RuntimeError('Do not understand epoch %s' % epoch)
        df.set_index(['trial', 'time',
                      'est_key', 'est_val'], inplace=True)
        df = rois.reduce(df)  # Reduce to visual clusters
        df.loc[:, 'sub'] = sub
        df.loc[:, 'session'] = session
        df.set_index(['sub', 'session'], append=True, inplace=True)
        sacc.append(df)
    sacc = pd.concat(sacc).reset_index()
    if trials is None:
        trials = slice(None)
    sacc = sacc.set_index('trial')
    cond = sacc.loc[trials, :].groupby(
        ['sub', 'time', 'est_key', 'est_val']).mean()
    return cond


