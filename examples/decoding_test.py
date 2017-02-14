from os import path
from scipy.io import loadmat
import pandas as pd
import numpy as np
import mne
from pymeg import preprocessing, artifacts
from conf_analysis.meg import decoding
from joblib import Memory
from os.path import expanduser, join
home = expanduser("~")
memory = Memory(cachedir=path.join(home, 'cache_pymeg'), verbose=0)

paths = {'c':'/home/pmurphy/Decoding_tests/Checker_location/Data/DC1/S1/Behaviour/',
         'go':'/home/pmurphy/Decoding_tests/Gabor_orientation/Data/DC1/S1/Behaviour/',
         'gl':'/home/pmurphy/Decoding_tests/Gabor_location/Data/DC1/S1/Behaviour/'}


c = lambda block: ('C', path.join(paths['c'], 'DC1_1_%i.mat'%block))
go = lambda block: ('GO', path.join(paths['go'], 'DC1_1_%i.mat'%block))
gl = lambda block: ('GL', path.join(paths['gl'], 'DC1_1_%i.mat'%block))


block_mapping = [c(1),  go(1), gl(1),
                 go(2), gl(2), c(2),
                 go(3), c(3),  gl(3)]

block_mapping_2 = [c(4), go(4), gl(4)]



def get_data():
    ea = get_epochs('/home/pmurphy/Decoding_tests/meg_data/DC1_TimeScale_20170201_01.ds')
    ea2 = get_epochs('/home/pmurphy/Decoding_tests/meg_data/DC1_TimeScale_20170201_02.ds')
    ea2.info['dev_head_t'] = ea.info['dev_head_t']
    ea, df = align(ea, epoch_offset=10)
    print df.shape
    ea2, df2 = align(ea2, epoch_offset=0, block_mapping=block_mapping_2)
    print df2.shape
    df2 = df2.set_index(2025+np.arange(len(df2)))
    meta = pd.concat([df, df2])
    epochs = mne.concatenate_epochs([ea, ea2])
    return epochs, meta

@memory.cache
def get_epochs(filename):
    raw = mne.io.read_raw_ctf(filename)
    raw.annotations =artifacts.annotate_blinks(raw, ch_mapping={'x':'UADC001-3705', 'y':'UADC003-3705', 'p':'UADC004-3705'})
    events = preprocessing.get_events(raw)[0]
    ide =  np.in1d(events[:, 2], [11, 31, 51])
    target_events = events[ide, :]
    epochs = mne.Epochs(raw, target_events, tmin=-0.2, tmax=0.9, baseline=(-0.1, 0), reject=dict(mag=4e-12))
    epochs.load_data()
    epochs.resample(300, n_jobs=4)
    return epochs


def get_meta(block_mapping=block_mapping):
    df = []
    index_offset = 0
    for i, (block, filename) in enumerate(block_mapping):
        data = loadmat(filename)['Behav'][:, 0]
        d = pd.DataFrame({'block':i, 'target':data, 'type':block})
        d = d.set_index(np.arange(250)+index_offset)
        if block=='GO':
            print 'Go'
            d.target-=90
        df.append(d)
        index_offset+=250
    return pd.concat(df)


def align(epochs, epoch_offset=0, block_mapping=block_mapping):
    df = get_meta(block_mapping)
    df = df.dropna()

    df = df.set_index(np.arange(len(df)))

    dropped = np.where(epochs.drop_log)[0]-epoch_offset
    assert (len(df) == len(epochs) + len(dropped) - epoch_offset)
    dropped = dropped[dropped>=0]
    if epoch_offset > 0:
        epochs.drop(np.arange(epoch_offset))
    df = df.drop(dropped)
    return epochs, df


def get_subset(epochs, df, labels):
    index = np.in1d(df.index.values, labels)
    return epochs[index], df.loc[labels, :]


def pairwise_decoding(epochs, meta, delta_target=15):
    # First get epochs.
    targets = np.unique(meta.target)
    targets = targets[targets>=(min(targets)+delta_target)]
    targets = targets[targets<=(max(targets)-delta_target)]
    results = []
    for it, target in enumerate(targets):
        predict_targets = targets[targets>=target][::-1]
        for jt, target2 in enumerate(predict_targets):
            index  = (
                     (
                        ((target-delta_target) < meta.target.values) &
                         (meta.target.values < (target+delta_target))
                     )
                        |
                     (
                        ((target2-delta_target) < meta.target.values) &
                         (meta.target.values < (target2+delta_target))
                     )
                )
            lselect = meta.loc[index].index.values
            ego, edf =  get_subset(epochs, meta, lselect)
            times = ego.times
            threshold = np.mean([target, target2])
            labels = (edf.target > threshold).values
            #print edf.target.values
            #print labels
            data = ego._data
            idx = (0.08<times) & (times<0.22)
            intm = []
            for id_time, time in zip(np.where(idx)[0], times[idx]):
                #print id_time, times[id_time]
                acc = decoding.decode(decoding.clf, data, labels, id_time, [id_time],
                        cv=decoding.cv, collapse=np.mean,
                        relabel_times=times)
                acc['t1'] = target
                acc['t2'] = target2
                intm.append(acc)
            results.append(pd.concat(intm))
    return results


def execute(x):
    mne.set_log_level('warning')
    epochs, meta = get_data()
    index = (meta.type==x).values
    ec = epochs[index]
    el = ec.pick_channels(decoding.sensors['occipital'](ec.ch_names))
    mc = meta.loc[index]
    acc = pairwise_decoding(el, mc)
    acc = pd.concat(acc)
    acc.to_hdf('decoding_pairwise_%s.hdf'%x, 'decoding')


def list_tasks(older_than='now'):
    for t in ['C', 'GL', 'GO']:
        yield t
