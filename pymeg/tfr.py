import glob
import mne
import locale
import pandas as pd
import numpy as np
import json

import h5py
from mne.externals import h5io
read = h5io._h5io._triage_read

from os.path import expanduser, join
home = expanduser("~")

from joblib import Memory

memory = Memory(cachedir=join(home, 'cache_pymeg'), verbose=0)

try:
    import seaborn as sns
    sns.set_style('ticks')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    pass


def taper_data(foi=None, cycles=None, time_bandwidth=None, **kwargs):
    if len(np.atleast_1d(cycles)) == 1:
        cycles = [cycles] * len(foi)
    foi = np.atleast_1d(foi)
    cycles = np.atleast_1d(cycles)
    time = cycles / foi
    f_smooth = time_bandwidth / time
    return zip(list(foi), list(cycles), list(time), list(f_smooth))


def describe_taper(foi=None, cycles=None, time_bandwidth=None, **kwargs):
    '''
    Print information about frequency smoothing / temporal smoothing for a set
    of taper parameters.
    '''
    from tabulate import tabulate
    data = taper_data(foi, cycles, time_bandwidth, **kwargs)
    print(tabulate(data,
                   headers=['Freq', 'Cycles', 't. window', 'F. smooth']))


def get_smoothing(F, foi=None, cycles=None, time_bandwidth=None, **kwargs):
    data = np.array(taper_data(foi, cycles, time_bandwidth, **kwargs))
    idx = np.argmin(np.abs(np.array(data[:, 0] - F)))
    return data[idx, 0], data[idx, 2], data[idx, 3]


def params_from_json(filename):
    '''
    Load taper parameters from a json file.
    '''
    params = json.load(open(filename))
    assert('foi' in params.keys())
    assert('cycles' in params.keys())
    assert('time_bandwidth' in params.keys())
    return params


def tfr(filename, outstr='tfr.hdf', foi=None, cycles=None,
        time_bandwidth=None, decim=10, n_jobs=4, **kwargs):
    '''
    Run TFR decomposition with multitapers.
    '''
    from mne.time_frequency.tfr import _tfr_aux

    outname = filename.replace('-epo.fif.gz', outstr)
    epochs = mne.read_epochs(filename)
    power = epochs_tfr(epochs, foi=foi, cycles=cycles,
                       time_bandwidth=time_bandwidth,
                       decim=decim, n_jobs=n_jobs, **kwargs)
    save_tfr(power, outname, epochs.events)
    return power


def epochs_tfr(epochs, foi=None, cycles=None, time_bandwidth=None,
               decim=10, n_jobs=4, **kwargs):
    from mne.time_frequency.tfr import _compute_tfr
    tfr_params = dict(n_cycles=cycles, n_jobs=n_jobs, use_fft=True,
                      zero_mean=True, time_bandwidth=time_bandwidth)
    power = _compute_tfr('multitaper', epochs, foi, decim, False, None, False,
                         output='complex', **tfr_params)
    return power


def tiling_plot(foi=None, cycles=None, time_bandwidth=None, **kwargs):
    colors = sns.cubehelix_palette(len(foi), light=0.75,  start=.5, rot=-.75)
    if len(np.atleast_1d(cycles)) == 1:
        cycles = [cycles] * len(foi)
    foi = np.atleast_1d(foi)
    cycles = np.atleast_1d(cycles)
    time = cycles / foi
    f_smooth = time_bandwidth / time

    plt.figure()
    currentAxis = plt.gca()
    for i, (f, t, w) in enumerate(zip(foi, time, f_smooth)):
        r = Rectangle((0 - (t / 2.), f - (w / 2.)), t, w, lw=2,
                      fill=None, alpha=.75, color=colors[i])
        currentAxis.add_patch(r)
    plt.ylim([foi[0] - f_smooth[0], foi[-1] + f_smooth[-1] / 1.5])
    plt.xlim([-time[0] / 1.5, time[0] / 1.5])
    plt.show()


def save_tfr(tfr, fname, events):
    import h5py
    with h5py.File(fname, 'w') as file:
        print('Saving new format')
        group = file.create_group('pymegtfr')
        group.attrs['freqs'] = tfr.freqs
        group.attrs['times'] = tfr.times
        group.attrs['channels'] = np.array(tfr.ch_names).astype('str')
        for event, trial in zip(events[:, 2], tfr.data):
            shape = trial.shape
            chunk_size = (shape[0], 1, 1)
            group.create_dataset(str(event), data=trial,
                                 chunks=chunk_size)


def get_tfrs(filenames, freq=(0, 100), channel=None, tmin=None, tmax=None,
             baseline=None):
    '''
    Load many saved tfrs and return as a data frame.

    Inputs
    ------
        filenames: List of TFR filenames
        freq:  tuple that specifies which frequencies to pull from TFR file.
        channel: List of channels to include
        tmin & tmax: which time points to include
        baseline: If func it will be applied to each TFR file that is being loaded.
    '''
    dfs = []
    for f in filenames:
        df = make_df(
            read_chunked_hdf(f, freq=freq, channel=channel, tmin=tmin, tmax=tmax))
        df.columns.name = 'time'
        if baseline is not None:
            df = baseline(df)
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.columns.name = 'time'
    return dfs


def read_chunked_hdf(fname, epochs=None, channel=None,
                     freq=(0, 150), tmin=0, tmax=1, key='pymegtfr'):
    with h5py.File(fname) as file:
        hdf = file[key]

        out = {}
        out['freqs'] = hdf.attrs['freqs']
        out['times'] = hdf.attrs['times']
        out['channels'] = hdf.attrs['channels']
        freq_idx = np.where((freq[0] <= out['freqs']) &
                            (out['freqs'] <= freq[1]))[0]
        freq_idx = slice(min(freq_idx), max(freq_idx) + 1)
        out['freqs'] = out['freqs'][freq_idx]
        if tmin is None:
            tmin = out['times'].min()
        if tmax is None:
            tmax = out['times'].max()
        time_idx = np.where((tmin <= out['times']) & (out['times'] <= tmax))[0]
        time_idx = slice(min(time_idx), max(time_idx) + 1)
        out['times'] = out['times'][time_idx]
        if channel is None:
            ch_id = slice(None)
        events, data = [], []
        if epochs is None:
            epochs = [int(i) for i in hdf.keys()]
        for epoch in epochs:
            data.append(hdf[str(epoch)][ch_id, freq_idx, time_idx])
            events.append(epoch)
        out['data'] = np.stack(data, 0)
        out['events'] = events
        out['channels'] = out['channels']
    return out


def make_df(out):
    freq = out['freqs']
    ch_ids = np.array(out['channels'])
    times = out['times']
    trials = out['events']
    trials, channel, freq = np.meshgrid(trials, ch_ids.ravel(),
                                        freq.ravel(),
                                        indexing='ij')
    index = pd.MultiIndex.from_arrays([trials.ravel(), channel.ravel(),
                                       freq.ravel()], names=['trial', 'channel', 'freq'])
    data = {}
    for t_idx, t in enumerate(times):
        data[t] = out['data'][:, :, :, t_idx].ravel()
    data = pd.DataFrame(data, index=index)
    data.columns.time = 'time'
    return data


@memory.cache
def read_info(hdf, fname):
    return read(hdf)
