import mne
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


def describe_taper(foi=None, cycles=None, time_bandwidth=None, **kwargs):
    '''
    Print information about frequency smoothing / temporal smoothing for a set
    of taper parameters.
    '''
    from tabulate import tabulate
    if len(np.atleast_1d(cycles))==1:
        cycles = [cycles]*len(foi)
    foi = np.atleast_1d(foi)
    cycles = np.atleast_1d(cycles)
    time = cycles/foi
    f_smooth = time_bandwidth/time
    data = zip(list(foi), list(cycles), list(time), list(f_smooth))
    print(tabulate(data,   headers=['Freq', 'Cycles', 't. window', 'F. smooth']))


def params_from_json(filename):
    '''
    Load taper parameters from a json file.
    '''
    params = json.load(open(filename))
    assert('foi' in params.keys())
    assert('cycles' in params.keys())
    assert('time_bandwidth' in params.keys())
    return params


def tfr(filename, outstr='tfr.hdf', foi=None, cycles=None, time_bandwidth=None, decim=10, n_jobs=4, **kwargs):
    '''
    Run TFR decomposition with multitapers.
    '''
    outname = filename.replace('-epo.fif.gz', outstr)
    epochs = mne.read_epochs(filename)
    power = mne.time_frequency.tfr_multitaper(epochs, foi, cycles,
        decim=decim, time_bandwidth=time_bandwidth, average=False, return_itc=False,
        n_jobs=n_jobs)
    save_tfr(power, outname, epochs.events)
    return power


def tiling_plot(foi=None, cycles=None, time_bandwidth=None, **kwargs):
    colors = sns.cubehelix_palette(len(foi), light=0.75,  start=.5, rot=-.75)
    if len(np.atleast_1d(cycles))==1:
        cycles = [cycles]*len(foi)
    foi = np.atleast_1d(foi)
    cycles = np.atleast_1d(cycles)
    time = cycles/foi
    f_smooth = time_bandwidth/time

    plt.figure()
    currentAxis = plt.gca()
    for i, ( f, t, w) in enumerate(zip(foi, time, f_smooth)):
        r = Rectangle((0 - (t/2.), f - (w/2.)), t, w, lw=2, fill=None, alpha=.75, color=colors[i])
        currentAxis.add_patch(r)
    plt.ylim([foi[0]-f_smooth[0], foi[-1]+f_smooth[-1]/1.5])
    plt.xlim([-time[0]/1.5, time[0]/1.5])
    plt.show()


def get_tfrs(filenames, freq=(0, 100), channel=None, tmin=None, tmax=None):
    '''
    Load many saved tfrs and return as a data frame.
    '''
    dfs = []
    for f in filenames:
        #try:
        df = make_df(read_hdf5(f, freq=freq, channel=channel, tmin=tmin, tmax=tmax))
        dfs.append(df)
        #except KeyError as e:
        #
    dfs =  pd.concat(dfs)
    dfs.columns.name = 'time'
    return dfs


def save_tfr(tfr, fname, events):
    from mne.externals import h5io
    h5io.write_hdf5(fname, {'data':tfr.data, 'freqs':tfr.freqs, 'times':tfr.times,
        'comment':tfr.comment, 'info':tfr.info, 'events':events}, overwrite=True)


def load_tfr(fname):
    from mne.externals import h5io
    data = h5io.read_hdf5(fname)#, freq=(-np.inf, np.inf), tmin=-np.inf, tmax=np.inf)
    events = data['events']
    del data['events']
    return mne.time_frequency.tfr.EpochsTFR(**data), events


def read_hdf5(fname, channel=None, freq=(0, 150), tmin=0, tmax=1, key='h5io'):
    hdf = h5py.File(fname)[key]
    keys = set(hdf.keys()) - set(['key_data', 'key_info'])
    out = {}
    for key in keys:
        out[key.replace('key_', '')] = read(hdf[key])
    out['info'] = read_info(hdf['key_info'], fname)
    freq_idx = np.where((freq[0] <= out['freqs']) & (out['freqs'] <= freq[1]))[0]
    freq_idx = slice(min(freq_idx), max(freq_idx)+1)
    out['freqs'] = out['freqs'][freq_idx]
    time_idx = np.where((tmin <= out['times']) & (out['times'] <= tmax))[0]
    time_idx = slice(min(time_idx), max(time_idx)+1)
    out['times'] = out['times'][time_idx]
    ch_names = np.array(out['info']['ch_names'])
    if channel is None:
        ch_id = np.arange(len(ch_names))
    else:
        try:
            channel = channel(ch_names)
        except TypeError:
            pass
        ch_id = np.where(np.in1d(ch_names, np.array(channel)))[0]
    out['data'] = hdf['key_data'][:, ch_id, freq_idx, time_idx]
    out['ch_id'] = ch_id
    return out


def make_df(out):
    freq = out['freqs']
    ch_ids = out['ch_id']
    times = out['times']
    trials = out['events'][:, 2]
    trials, channel, freq = np.meshgrid(trials, ch_ids.ravel(),
                                              freq.ravel(),
                                              indexing='ij')
    index = pd.MultiIndex.from_arrays([trials.ravel(), channel.ravel(),
        freq.ravel()], names=['trial', 'channel', 'freq'])
    data = {}
    for t_idx, t in enumerate(times):
        data[t] = out['data'][:,:,:,t_idx].ravel()
    data = pd.DataFrame(data, index=index)
    data.columns.time = 'time'
    return data


@memory.cache
def read_info(hdf, fname):
    return read(hdf)
