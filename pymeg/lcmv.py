from __future__ import division
from __future__ import print_function

import logging
import os

from itertools import product

import numpy as np
import pandas as pd

from joblib import Memory
from joblib import Parallel, delayed

from mne import compute_covariance
from mne.beamformer import make_lcmv
#from mne.beamformer._lcmv import _apply_lcmv
from mne.time_frequency.tfr import _compute_tfr

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)

fois = np.arange(10, 150, 5)
default_tfr = {'foi': fois, 'cycles': fois * 0.1, 'time_bandwidth': 2,
               'n_jobs': 1, 'est_val': fois, 'est_key': 'F'}


def power_est(x, time, est_val=None, est_key=None, sf=600., foi=None,
              cycles=None, time_bandwidth=None, n_jobs=1, decim=10):
    '''
    Estimate power of epochs in array x.
    '''
    if len(x.shape) == 2:
        x = x[np.newaxis, :, :]
    y = _compute_tfr(
        x, foi, sfreq=sf, method='multitaper', decim=decim, n_cycles=cycles,
        zero_mean=True, time_bandwidth=time_bandwidth, n_jobs=n_jobs,
        use_fft=True, output='complex')
    return y, time[::decim], est_val, est_key


def broadband_est(x, time, est_val=[-1], est_key='BB', **kwargs):
    return x, time, est_val, est_key


def tfr2power_estimator(x):
    return ((x * x.conj()).real).mean(1)


def accumulate(data, time, est_key, est_val, roi, trial):
    '''
    Transform SR results to data frame.

    Arguments
    ---------

    data: ndarray
        If ntrials x vertices x time in which case the
        function will average across vertices.
        If ntrials x time will be directly converted to df.
    time: ndarray
        time points that match last dimension of data
    est_key: value
        estimation key for this value
    est_val: value
        estimation value for this set of data
    roi: str
        Name of the roi that this comes from
    trial: ndarray
        Needs to match first dim of data
    '''
    if data.ndim == 3:
        data = flip_vertices(data).mean(1)

    # Now ntrials x time
    df = pd.DataFrame(data, index=trial, columns=time)
    df.columns.name = 'time'
    df.index.name = 'trial'
    df = df.stack().reset_index()
    df.loc[:, 'est_key'] = est_key
    df.loc[:, 'est_val'] = est_val
    df.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    df.columns = [roi]
    return df


def flip_vertices(data):
    '''
    Average over vertices but correct for random flips first
    Correction is done by ensuring positive correlations
    between vertices
    '''
    for i in range(data.shape[1]):
        if np.corrcoef(data[:, i, :].ravel(),
                       data[:, 0, :].ravel())[0, 1] < 0:
            data[:, i, :] = -data[:, i, :]
    if all(data.mean((0, 2)) < 0):
        data = -data
    return data


@memory.cache
def setup_filters(info, forward, data_cov, noise_cov, labels,
                  reg=0.05, pick_ori='max-power', njobs=4):
    logging.info('Getting filter')
    tasks = []
    for l in labels:
        tasks.append(delayed(get_filter)(
            info, forward, data_cov,
            noise_cov, label=l, reg=reg,
            pick_ori='max-power'))

    filters = Parallel(n_jobs=njobs, verbose=1)(tasks)
    return {name: f for name, f in filters}


def reconstruct_tfr(
        filters, info, epochs,  events, times,
        estimator=power_est, est_args=default_tfr,
        post_func=tfr2power_estimator, accumulate_func=accumulate,
        njobs=4):
    '''
    Reconstruct a set of epochs with filters based on data_cov and forward
    model.
    '''
    M = par_reconstruct(
        pre_estimator=estimator, pre_est_args=est_args, epochs=epochs,
        events=events, times=times, info=info, filters=filters,
        post_func=post_func, accumulate_func=accumulate_func, njobs=njobs)
    return pd.concat([pd.concat(m, axis=0) for m in M], axis=1)
    # return pd.concat(M, axis=1)


def reconstruct_broadband(
        filters, info, epochs,  events, times,
        estimator=broadband_est, est_args={}, post_func=None,
        accumulate_func=accumulate, njobs=4):
    '''
    Reconstruct a set of epochs with filters based on data_cov and forward
    model.
    '''
    M = par_reconstruct(
        pre_estimator=estimator, pre_est_args=est_args, epochs=epochs,
        events=events, times=times, info=info, filters=filters,
        post_func=None, accumulate_func=accumulate_func, njobs=njobs)

    return pd.concat([pd.concat(m, axis=0) for m in M], axis=1)


def par_reconstruct(pre_estimator, pre_est_args, epochs, events, times,
                    info, filters, post_func=tfr2power_estimator,
                    accumulate_func=accumulate, njobs=4):
    '''
    Apply LCMV source reconstruction to epochs and apply a filter before and
    after.
    '''
    pre_est_args['n_jobs'] = njobs
    logging.info('Applying pre-estimator function, params: ' +
                 str(pre_est_args))
    tfrdata, times, est_val, est_key = pre_estimator(
        epochs, times, **pre_est_args)
    logging.info('Done with pre-estimator. Data has shape ' +
                 str(tfrdata.shape) + ' now')
    tasks = []

    for filter in filters.keys():
        tasks.append(delayed(apply_lcmv)(
            tfrdata, est_key, est_val, events,
            times, info,
            {filter: filters[filter]}, post_func=post_func,
            accumulate_func=accumulate_func))
    logging.info(
        'Prepared %i tasks for parallel execution with %i jobs' %
        (len(tasks), njobs)
    )
    return Parallel(n_jobs=njobs, verbose=1)(tasks)


def apply_lcmv(tfrdata, est_key, est_vals, events, times, info,
               filters, post_func=None, accumulate_func=None,
               max_ori_out='signed'):
    '''
    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.


    Arguments
    ---------
    tfrdata: ndarray
        Data to be reconstructed.
        Should be either n_trials x n_sensors x Y x n_time
        or trials x sensors x time. Reconstruction treats epochs and
        dim Y as independent dimensions.
    est_key: value
        A key to identify this reconstruction (e.g. F for power)
    est_vals: sequence
        Values that identify different reconstructions along dimension Y
        for a single epoch, e.g. the frequency for power reconstructions.
        Needs to be length Y.
    events: array
        Identifiers for different epochs. Needs to be of length n_trials.
    times: array
        Time of entries in last dimension of input data.
    info: mne info structure
        Info structure of the epochs which are to be reconstructed
    filters: dict
        Contains ROI names as keys and MNE filter dicts as values.
    post_func: function
        This function is applied to the reconstructed epochs, useful
        to convert complex TFR estimates into power values.
    accumulate_func: function
        Function that is applied after post_func has been applied.
        Can for example be used to transform the output into a dataframe.
    max_ori_out: str, default 'signed'
        This is passed to the MNE LCMV function which at the moment requires
        this to be 'signed'

    Output
    ------
        List of source reconstructed epochs transformed by post_func.
    '''

    if accumulate_func is None:
        accumulate_func = lambda x: x
    if tfrdata.ndim == 3:
        # Must be trials x n_sensors x t_time
        tfrdata = tfrdata[:, :, np.newaxis, :]
    nfreqs = tfrdata.shape[2]
    assert(len(est_vals) == nfreqs)
    # ntrials = tfrdata.shape[0]
    info['sfreq'] = 1. / np.diff(times)[0]
    results = []
    for freq, (roi, filter) in product(range(nfreqs), filters.items()):
        if filter['weights'].size > 0:
            data = np.stack([x._data for x in
                             _apply_lcmv(data=tfrdata[:, :, freq, :],
                                         filters=filter,
                                         info=info, tmin=times.min(),
                                         max_ori_out=max_ori_out)])
            if post_func is None:
                results.append(accumulate_func(
                    data, est_key=est_key, time=times, est_val=est_vals[freq],
                    roi=roi, trial=events))
            else:
                data = post_func(data)
                results.append(accumulate_func(
                    data, est_key=est_key, time=times,
                    est_val=est_vals[freq], roi=roi, trial=events))
    return results


@memory.cache
def get_cov(epochs, tmin=0, tmax=1):
    return compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk')


@memory.cache
def get_noise_cov(epochs, tmin=-0.5, tmax=0):
    return compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk')


def get_filter(info, forward, data_cov, noise_cov, label=None, reg=0.05,
               pick_ori='max-power'):
    filter = make_lcmv(info=info,
                       forward=forward,
                       data_cov=data_cov,
                       noise_cov=noise_cov,
                       reg=0.05,
                       pick_ori='max-power',
                       label=label)
    del filter['data_cov']
    del filter['noise_cov']
    del filter['src']
    return label.name, filter


def get_filters(estimator, epochs, forward, source, noise_cov, data_cov,
                labels):
    return {l.name: get_filter(epochs.info, forward, data_cov, noise_cov,
                               label=l, reg=0.05, pick_ori='max-power')
            for l in labels}


def _apply_lcmv(data, filters, info, tmin, max_ori_out):
    """Apply LCMV spatial filter to data for source reconstruction.
    Copied directly from MNE to remove dependence on source space in
    filter. This makes the filter much smaller and easier to use 
    multiprocessing here.

    Original authors: 
    Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
              Roman Goj <roman.goj@gmail.com>
              Britta Westner <britta.wstnr@gmail.com>

    Original License: BSD (3-clause)
    """
    from mne.source_estimate import _make_stc
    from mne.minimum_norm.inverse import combine_xyz
    if max_ori_out != 'signed':
        raise ValueError('max_ori_out must be "signed", got %s'
                         % (max_ori_out,))

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    W = filters['weights']

    #subject = _subject_from_forward(filters)
    for i, M in enumerate(data):
        if len(M) != len(filters['ch_names']):
            raise ValueError('data and picks must have the same length')

        if filters['is_ssp']:
            raise RuntimeError('SSP not supported here')

        if filters['whitener'] is not None:
            M = np.dot(filters['whitener'], M)

        # project to source space using beamformer weights
        vector = False
        if filters['is_free_ori']:
            sol = np.dot(W, M)
            if filters['pick_ori'] == 'vector':
                vector = True
            else:
                sol = combine_xyz(sol)
        else:
            # Linear inverse: do computation here or delayed
            if (M.shape[0] < W.shape[0] and
                    filters['pick_ori'] != 'max-power'):
                sol = (W, M)
            else:
                sol = np.dot(W, M)
            if filters['pick_ori'] == 'max-power' and max_ori_out == 'abs':
                sol = np.abs(sol)

        tstep = 1.0 / info['sfreq']
        yield _make_stc(sol, vertices=filters['vertices'], tmin=tmin,
                        tstep=tstep, subject='NN', vector=vector,
                        source_nn=filters['source_nn'])
