'''
%% TRIGGERS

trigger.address      = hex2dec('378');
trigger.zero         = 0;
trigger.width        = 0.005; %1 ms trigger signal

%Target responses, left vs right by correct vs error, garbage
trigger.leftcor    = 20;
trigger.rightcor = 21;
trigger.lefterr    = 22;
trigger.righterr   = 23;

trigger.garbageresp     = 29;

%confidence on
trigger.cjon = 32;
%confidece judgment
trigger.cjoff = [11:17]; %11:16 mapping to 1:6, 17 mapping onto garbage

trigger.startblock = 32;
trigger.endblock = 33;

%motion triggers
trigger.rndmotion = 41; 
trigger.cohm = 42;
trigger.irimotion = 43; 

'''
import pickle
import mne
import numpy as np
from pymeg.preprocessing import get_meta, preprocess_block, get_epoch

savepath = '/home/nwiming/'

mapping = {20: ('response', 20),
           21: ['response', 21],
           22: ['response', 22],
           23: ['response', 23],
           29: ['response', 29],
           32: ['confidence_onset', 0],
           11: ['confidence', 1],
           12: ['confidence', 2],
           13: ['confidence', 3],
           14: ['confidence', 4],
           15: ['confidence', 5],
           16: ['confidence', 6],
           41: ['motion_on', 1],
           42: ['coherence_on', 1],
           43: ['irimotion', 1]}


def to_blocks(timing, sfreq=1200., min_dur=8, max_dur=12):
    '''
    Find breaks to cut data into pieces.

    Returns a dictionary ob blocks with a namedtuple that
    defines start and end of a block. Start and endpoint
    are inclusive (!).    
    '''
    from collections import namedtuple
    onsets = (timing.values - timing.values[0]) / sfreq / 60
    diffs = np.diff(onsets)
    id_break = np.where(diffs > 1)[0]
    bounds = [-1] + list(id_break) + [len(timing) - 1]
    Block = namedtuple('Block', ['start', 'end', 'start_trial', 'end_trial'])
    blocks = {block: Block(timing.iloc[start + 1], timing.iloc[end],
                           start + 1, end) for block, (start, end)
              in enumerate(zip(bounds[:-1], bounds[1:]))}
    for start, end in blocks.values():
        dur = (onsets[end] - onsets[start])
        assert(min_dur < dur)
        assert(dur < max_dur)
    return blocks


def filenames(subject, epoch, recording, block):
    from os.path import join
    path = join(savepath, 'seqconf')
    fname = 'S%02i_%s_R%i_B%i' % (subject, epoch, recording, block)
    return (join(path, fname + '-fif.gz'),
            join(path, fname + '.meta'),
            join(path, fname + '.artdef'))


def preprocess_raw(subject, recording, filename):
    raw = mne.io.ctf.read_raw_ctf(
        '/home/kdesender/meg_data/seqconf/Pilot01-01_Seqconf_20190123_01.ds')

    meta, timing = get_meta(raw, mapping, {}, 41, 41)

    blocks = to_blocks(timing.coherence_on)

    min_start, max_end = np.min(raw.times), np.max(raw.times)

    for i, block in blocks.values():
        # Cut into blocks
        start, end = raw.times[block.start], raw.times[block.end]
        start = np.maximum(start, min_start)
        end = np.minimum(end, max_end)
        r = raw.copy().crop(start, end)
        start, end = block.start, block.end

        r, ants, artdef = preprocess_block(raw)

        print('Notch filtering')
        midx = np.where([x.startswith('M') for x in r.ch_names])[0]
        r.notch_filter(np.arange(50, 251, 50), picks=midx)

        block_meta = meta.loc[block.start_trial:block.end_trial + 1, :]
        block_timing = timing.loc[block.start_trial:block.end_trial + 1, :]

        # Cut into epochs
        for epoch, event, (tmin, tmax), (rmin, rmax) in zip(
                ['stimulus', 'response'],
                ['coherence_on_time', 'response_time'],
                [(-1, 3), (-1.5, 1.5)],
                [(-0.5, 2.5), (-1.5, 1)]):

            m, s = get_epoch(r, block_meta, block_timing,
                             event=event, epoch_time=(
                                 tmin, tmax),
                             reject_time=(rmin, rmax),
                             )

            if len(s) <= 0:
                continue
            epofname, mfname, afname = filenames(subject, epoch, recording, i)
            s = s.resample(600, npad='auto')
            s.save(epofname)
            m.to_hdf(mfname, 'meta')
            pickle.dump(artdef, open(afname, 'wb'), protocol=2)
