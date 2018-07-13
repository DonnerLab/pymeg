# Plotting
# matplotlib inline
import os
from joblib import Memory
import glob

import h5py
# Basics

import numpy as np

# MNE
import mne
from mne import create_info
from mne.epochs import EpochsArray

mne.set_log_level('WARNING')


memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)


def fix_chs(rawinfo, einfo):
    for k in range(len(einfo['chs'])):
        name = einfo['chs'][k]['ch_name']
        newchan = [x for x in rawinfo['chs'] if name in x['ch_name']][0]
        #newchan['ch_name'] = newchan['ch_name'].replace('-3705', '')
        einfo['chs'][k] = newchan
    return einfo


@memory.cache
def get_info_for_epochs(rawname):
    raw = mne.io.ctf.read_raw_ctf(rawname)
    return raw.info

# os.chdir("P01/MEG/TFR")


def load_ft_epochs(fname, rawinfo):
    # load Matlab/Fieldtrip data
    f = h5py.File(fname)
    list(f.keys())
    ft_data = f['data']
    ft_data.keys()

    trialinfo = ft_data['trialinfo']
    channels = ft_data['label']
    sampleinfo = ft_data['sampleinfo']
    time = ft_data['time']
    sfreq = 1 / np.diff(time)
    assert(len(np.unique(sfreq)) == 1)
    n_time, n_chans, n_trial = ft_data['trial'].shape

    data = np.zeros((n_trial, n_chans, n_time))
    transposed_data = np.transpose(ft_data['trial'])
    for trial in range(n_trial):
        data[trial, :, :] = transposed_data[trial]
    data = data[:, range(n_chans), :]
    chan_names = []
    for i in range(n_chans):
        st = channels[0][i]
        obj = ft_data[st]
        chan_names.append(''.join(chr(j) for j in obj[:]))
    ch_names = [x + '-3705' for x in chan_names]
    info = create_info(ch_names, sfreq)
    events = np.zeros((n_trial, 3), int)
    events[:, 2] = trialinfo[21]
    events[:, 0] = sampleinfo[0]

    epochs = EpochsArray(data, info, tmin=time.min(),
                         events=events, verbose=False)

    fix_chs(rawinfo, epochs.info)
    return epochs


if __name__ == "__main__":
    # , 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P22', 'P23', 'P24', 'P25', 'P28', 'P29', 'P31', 'P33']
    subjects = ['P01']

    for s in subjects:
        #os.chdir('/' + str(s) + "/MEG/TFR/")
        for file in sorted(glob.glob(str(s) + '/MEG/Locked/' + str(s) + '*rec*_stim.mat')):
            session = str(file[20])
            recording = str(file[25])
     #       rawfile = glob.glob(str(s) + '/MEG/Raw/' + '*-0' + str(session) + '*_0' + str(recording) + '.ds')
            rawfile = glob.glob(str(s) + '/MEG/Raw/' + '*-' +
                                str(session) + '*_0' + str(recording) + '.ds')
            rawinfo = get_info_for_epochs(rawfile[0])
            epochs = load_ft_epochs(file, rawinfo)
            epochs.save(str(file[:-4]) + '-epo.fif')
    #    epochs = load_ft_epochs(fname)
