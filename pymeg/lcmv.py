from __future__ import division
from __future__ import print_function

import mne
import numpy as np
import pandas as pd
import os

from . import source_reconstruction as sr
from joblib import Memory

from mne import compute_covariance
from mne.beamformer import lcmv_epochs

from . import tfr

from itertools import izip


memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)


@memory.cache
def get_cov(epochs, tmin=0, tmax=1):
    return compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk')


@memory.cache
def get_noise_cov(epochs, tmin=-0.5, tmax=0):
    return compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk')


def reconstruct_and_save(subject,
                         raw_filename, epochs_filename, trans_filename,
                         epochs,
                         srfilename, lcmvfilename,
                         debug=False):
    '''
    Example function for how to call reconstruct.

    Arguments
    ---------
    subject : Subject identifier
    raw_filename, epochs_filename, trans_filename : str
        These are passed along to source_reconstruct.get_leadfield
    epochs : MNE Epochs object
        The epochs to source reconstruct
    srfilename : str
        Where to save average response
    lcmvfilename : str
        Where to save source reconstructed data

    Returns
    -------
        None
    '''
    mne.set_log_level('WARNING')

    estimators = (get_broadband_estimator(),
                  get_highF_estimator(),
                  get_lowF_estimator())
    accumulator = AccumSR(srfilename)
    cov = get_cov(epochs)
    forward, bem, source, trans = sr.get_leadfield(
        subject, raw_filename, epochs_filename, trans_filename)
    labels = sr.get_labels(subject)
    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])

    if debug:
        # Select only 2 trials to make debuggin easier
        epochs = epochs[[str(l) for l in epochs.events[:2, 2]]]

    source_epochs = reconstruct(
        epochs=epochs,
        forward=forward,
        source=source,
        noise_cov=None,
        data_cov=cov,
        labels=labels,
        func=estimators,
        accumulator=accumulator)

    source_epochs.to_hdf(lcmvfilename, 'epochs')
    accumulator.save_averaged_sr()


def reconstruct(epochs, forward, source, noise_cov, data_cov, labels,
                func=None, accumulator=None):
    '''
    Perform SR for a set of epochs.

    Arguments
    ---------
        epochs : MNE Epochs object
        forward : MNE forward solution
        source : MNE source space
        noise_cov : MNE noise covariance matrix
        data_cov : MNE data covariance matrix
        labels : List of freesurfer labels
        func : None or list of function tuples
            func can apply a function to the source-reconstructed data, for
            example to perform a time-frqeuncy decomposition. The behavior of
            reconstruct  depends on what this function returns. If it returns
            a 2D matrix, it  will be interpreted as source_locations*num_F*time
            array. Importantly these functions will be applied before
            extraction of labels and the corresponding averaging operation.
            That is, passing in TFR functions here will perform TFR on all
            vertices and only afterwards will power be averaged. func itself
            should be a list of  tuples:  ('identifier', identifier values,
            function). This allows to label the transformed data appropriately.
            The resulting data frame will label each reconstructed value with
            identifier and the corresponding identifier value. If the function
            returns a 2D matrix, then identifier values needs to match the
            corresponding dimension (e.g. be the number of frequencies for
            TFR).
        accumulator : AccumSR object
    '''
    results = []
    if labels is None:
        labels = []
    index = epochs.events[:, 2]
    if not (np.unique(index) == len(epochs.events.shape[0])):
        index = np.arange(epochs._data.shape[0])

    for trial, epoch in izip(index,
                             lcmv_epochs(epochs, forward, noise_cov, data_cov,
                                         reg=0.05,
                                         pick_ori='max-power',
                                         return_generator=True)):
        if func is None:
            srcepoch = extract_labels_from_trial(
                epoch, labels, int(trial), source)
            results.append(srcepoch)
            if not accumulator is None:
                accumulator.update(epoch)
            del epoch
        else:
            for keyname, values, function in func:
                print('Running', keyname, 'on trial', int(trial))
                transformed = function(epoch.data)

                tstep = epoch.tstep / \
                    (float(transformed.shape[2]) / len(epoch.times))

                for value, row in zip(values, np.arange(transformed.shape[1])):

                    new_epoch = mne.SourceEstimate(transformed[:, row, :],
                                                   vertices=epoch.vertices,
                                                   tmin=epoch.tmin,
                                                   tstep=tstep,
                                                   subject=epoch.subject)

                    srcepoch = extract_labels_from_trial(
                        new_epoch, labels, int(trial), source)
                    srcepoch['est_val'] = value
                    srcepoch['est_key'] = keyname
                    results.append(srcepoch)
                    if not accumulator is None:
                        accumulator.update(keyname, value, new_epoch)
                    del new_epoch
                del transformed
            del epoch

    if len(labels) > 0:
        results = pd.concat([to_df(r) for r in results])
    else:
        results = None
    return results


def extract_labels_from_trial(epoch, labels, trial, source):
    srcepoch = {'time': epoch.times, 'trial': trial}
    for label in labels:
        try:
            pca = epoch.extract_label_time_course(
                label, source, mode='mean')
        except ValueError:
            pass
            # print('Source space contains no vertices for', label)
        srcepoch[label.name] = pca
    return srcepoch


class AccumSR(object):
    '''
    Accumulate SRs and compute an average.
    '''

    def __init__(self, filename, keyname, value):
        self.stc = None
        self.N = 0
        self.filename = filename
        self.keyname = keyname
        self.value = value

    def update(self, keyname, value, stc):
        if (self.keyname == keyname) and (self.value == value):
            if self.stc is None:
                self.stc = stc.copy()
            else:
                self.stc += stc
            self.N += 1

    def save_averaged_sr(self):
        stcs = self.stc.copy()
        idbase = (-.5 < stcs.times) & (stcs.times < 0)
        m = stcs.data[:, idbase].mean(1)[:, np.newaxis]
        s = stcs.data[:, idbase].std(1)[:, np.newaxis]
        stcs.data = (stcs.data - m) / s
        stcs.save(self.filename)
        return stcs


def get_highF_estimator(sf=600, decim=10):
    fois = np.arange(10, 151, 5)
    cycles = 0.1 * fois
    tb = 2
    return ('F', fois, get_power_estimator(fois, cycles, tb, sf=sf,
                                           decim=decim))


def get_lowF_estimator(sf=600, decim=10):
    fois = np.arange(1, 21, 2)
    cycles = 0.25 * fois
    tb = 2
    return ('LF', fois, get_power_estimator(fois, cycles, tb, sf=sf,
                                            decim=decim))


def get_broadband_estimator():
    return ('BB', [-1], lambda x: x[:, np.newaxis, :])


def get_power_estimator(F, cycles, time_bandwidth, sf=600., decim=1):
    '''
    Estimate power from source reconstruction

    This will return a num_trials*num_F*time array
    '''
    import functools

    def foo(x, sf=600.,
            foi=None,
            cycles=None,
            time_bandwidth=None,
            n_jobs=None, decim=decim):
        x = x[np.newaxis, :, :]
        x = tfr.array_tfr(x,
                          sf=sf,
                          foi=foi,
                          cycles=cycles,
                          time_bandwidth=time_bandwidth,
                          n_jobs=4, decim=decim)
        return x.squeeze()

    return functools.partial(foo, sf=sf,
                             foi=F,
                             cycles=cycles,
                             time_bandwidth=time_bandwidth,
                             n_jobs=4, decim=decim)


def to_df(r):
    length = len(r['time'])
    p = {}

    for key in r.keys():
        try:
            p[key] = r[key].ravel()
            if len(p[key]) == 1:
                p[key] = [r[key]] * length
        except AttributeError:
            p[key] = [r[key]] * length
    return pd.DataFrame(p)
