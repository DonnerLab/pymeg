import sys
sys.path.append('/home/nwilming/')
import glob
import mne, locale
import pandas as pd
import numpy as np
import cPickle
import json

try:
    import seaborn as sns
    sns.set_style('ticks')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    pass


def describe_taper(foi=None, cycles=None, time_bandwidth=None, **kwargs):
    from tabulate import tabulate
    if len(np.atleast_1d(cycles))==1:
        cycles = [cycles]*len(foi)
    foi = np.atleast_1d(foi)
    cycles = np.atleast_1d(cycles)
    time = cycles/foi
    f_smooth = time_bandwidth/time
    data = zip(list(foi), list(cycles), list(time), list(f_smooth))
    print tabulate(data,   headers=['Freq', 'Cycles', 't. window', 'F. smooth'])


def params_from_json(filename):
    params = json.load(open(filename))
    assert('foi' in params.keys())
    assert('cycles' in params.keys())
    assert('time_bandwidth' in params.keys())
    return params


def tfr(filename, outstr='tfr.hdf', foi=None, cycles=None, time_bandwidth=None, decim=10, n_jobs=12, **kwargs):
    outname = filename.replace('-epo.fif.gz', outstr)
    epochs = mne.read_epochs(filename)
    power = mne.time_frequency.tfr_multitaper(epochs, foi, cycles,
        decim=decim, time_bandwidth=time_bandwidth, average=False, return_itc=False,
        n_jobs=12)
    print filename, '-->', outname
    save_tfr(power, outname)
    #cPickle.dump({'power': power,
    #              'foi': foi,
    #              'cycles': cycles,
    #              'time_bandwidth': time_bandwidth,
    #              'decim':decim,
    #              'events':epochs.events}, open(outname, 'w'), 2)
    return True


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



def combine_tfr(filename, freq=(0, 100), channel=None, tmin=None, tmax=None):
    tfr = cPickle.load(open(filename))
    foi = tfr['power'].freqs
    foi = foi[(freq[0] < foi) & (foi <freq[1])]
    if channel is None:
        channel = tfr['power'].ch_names
    return tfr2df(tfr['power'], foi, channel, tmin=tmin, tmax=tmax, hash=tfr['events'][:,2])


def from_pickle(filename, freq=(0, 100), channel=None, tmin=None, tmax=None):
    tfr = cPickle.load(open(filename))
    foi = tfr['power'].freqs
    foi = foi[(freq[0] < foi) & (foi <freq[1])]
    if channel is None:
        channel = tfr['power'].ch_names
    return tfr2df(tfr['power'], foi, channel, tmin=tmin, tmax=tmax, hash=tfr['events'][:,2])


def tfr2df(tfr, freq, channel, tmin=None, tmax=None, hash=None):
    '''
    Read out values for specific frequencies and channels from set of tfrs.

    channels is a list of channel names.
    freq is a list of frequencies
    '''
    try:
        channel = channel(tfr.ch_names)
    except:
        pass
    freq = np.atleast_1d(freq)
    if tmin is not None and tmax is not None:
        tfr.crop(tmin, tmax)
    ch_ids = np.where(np.in1d(tfr.ch_names, channel))[0]

    ch_idx = np.in1d(np.arange(tfr.data.shape[1]), ch_ids)
    freq_idx = np.in1d(tfr.freqs, freq)
    tfr.data = tfr.data[:, ch_ids, :,:][:, :, np.where(freq_idx)[0], :]
    print tfr.data.shape
    tfr.freqs = tfr.freqs[freq_idx]
    if hash is None:
        trials = np.arange(tfr.data.shape[0])
    else:
        trials = hash

    trials, channel, freq, time = np.meshgrid(trials, ch_ids.ravel(),
                                              tfr.freqs.ravel(), tfr.times.ravel(),
                                              indexing='ij')
    assert trials.shape==tfr.data.shape
    return pd.DataFrame({'trial':trials.ravel(), 'channel':channel.ravel(),
        'freq':freq.ravel(), 'time':time.ravel(), 'power':tfr.data.ravel()})


def save_tfr(tfr, fname):
    from mne.externals import h5io
    h5io.write_hdf5(fname, {'data':tfr.data, 'freqs':tfr.freqs, 'times':tfr.times,
        'comment':tfr.comment, 'info':tfr.info}, overwrite=True)


def load_tfr(fname):
    return mne.time_frequency.tfr.EpochsTFR(**h5io.read_hdf5(fname))
