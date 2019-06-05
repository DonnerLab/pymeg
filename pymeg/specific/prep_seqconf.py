'''
TRIGGERS

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
from collections import namedtuple
from os.path import join
from pymeg.preprocessing import get_meta, preprocess_block, get_epoch, get_events

inpath = '/home/kdesender/meg_data/seqconf/'
savepath = '/home/nwilming/seqconf'

Recording = namedtuple('Recording', ["filename", "subject", "session", "block"])

recordings = [
  #Recordings     
  Recording('S3-1_Seqconf_20190323_01.ds', 3, 1, [1,2,3,4]),
  Recording('S3-1_Seqconf_20190323_02.ds', 3, 1, [5,6,7,8]),
  Recording('S3-2_Seqconf_20190324_01.ds', 3, 2, [1,2,3,4]),
  Recording('S3-2_Seqconf_20190324_02.ds', 3, 2, [5,6,7,8]), 
  Recording('S04-1_Seqconf_20190312_01.ds', 4, 1, [1,2,3,4]),
  Recording('S04-1_Seqconf_20190312_02.ds', 4, 1, [5,6,7,8]),
  Recording('S04-2_Seqconf_20190317_01.ds', 4, 2, [1,2,3,4]),
  Recording('S04-2_Seqconf_20190317_02.ds', 4, 2, [5,6,7,8]),
  Recording('S05-1_Seqconf_20190316_02.ds', 5, 1, [1,2]),
  Recording('S05-1_Seqconf_20190316_03.ds', 5, 1, [3,4]),
  Recording('S05-1_Seqconf_20190316_04.ds', 5, 1, [5,6]),
  Recording('S05-1_Seqconf_20190316_05.ds', 5, 1, [7,8]),
  Recording('S05-2_Seqconf_20190317_01.ds', 5, 2, [1,2,3,4]),
  Recording('S05-2_Seqconf_20190317_02.ds', 5, 2, [5,6,7,8]),
  Recording('S06-1_Seqconf_20190316_01.ds', 6, 1, [1,2,3,4]),
  Recording('S06-1_Seqconf_20190316_02.ds', 6, 1, [5,6,7,8]),
  Recording('S06-2_Seqconf_20190317_01.ds', 6, 2, [1,2,3,4]),
  Recording('S06-2_Seqconf_20190317_02.ds', 6, 2, [5,6,7,8]),
  Recording('S07-1_Seqconf_20190321_01.ds', 7, 1, [1,2,3,4]),
  Recording('S07-1_Seqconf_20190321_02.ds', 7, 1, [5,6,7, 8]),
  Recording('S07-2_Seqconf_20190326_01.ds', 7, 2, [9,10,11]),
  Recording('S07-2_Seqconf_20190326_02.ds', 7, 2, [5,6,7,8]),  
  Recording('S08-1_Seqconf_20190321_01.ds', 8, 1, [9,10]),
  Recording('S08-1_Seqconf_20190321_02.ds', 8, 1, [5,6,7,8]),
  Recording('S08-2_Seqconf_20190326_01.ds', 8, 2, [1,2,3,4]),
  Recording('S08-2_Seqconf_20190326_02.ds', 8, 2, [5,6,7,8]),
  Recording('S09-1_Seqconf_20190323_01.ds', 9, 1, [1,2,3,4]),
  Recording('S09-1_Seqconf_20190323_02.ds', 9, 1, [5,6,7,8]),
  Recording('S09-2_Seqconf_20190328_01.ds', 9, 2, [1,2,3,4]),
  Recording('S09-2_Seqconf_20190328_02.ds', 9, 2, [5,6,7,8]),
  Recording('S10-1_Seqconf_20190327_01.ds', 10, 1, [1,2,3,4]),
  Recording('S10-1_Seqconf_20190327_02.ds', 10, 1, [5,6,7,8]),
  Recording('S10-2_Seqconf_20190328_01.ds', 10, 2, [1,2,3,4]),
  Recording('S10-2_Seqconf_20190328_02.ds', 10, 2, [5,6,7,8]),
  Recording('S11-1_Seqconf_20190321_01.ds', 11, 1, [1,2,3,4]),
  Recording('S11-1_Seqconf_20190321_02.ds', 11, 1, [5,6,7,8]),
  Recording('S11-2_Seqconf_20190327_01.ds', 11, 2, [1,2,3,4]),
  Recording('S11-2_Seqconf_20190327_02.ds', 11, 2, [5,6,7,8]),
  Recording('S12-1_Seqconf_20190323_01.ds', 12, 1, [1,2,3,4]),
  Recording('S12-1_Seqconf_20190323_02.ds', 12, 1, [5,6,7,8]),
  Recording('S12-2_Seqconf_20190324_01.ds', 12, 2, [1,2,3,4]),
  Recording('S12-2_Seqconf_20190324_02.ds', 12, 2, [5,6,7,8]),
  Recording('S12-2_Seqconf_20190324_03.ds', 12, 3, [9,10]),
  Recording('S13-1_Seqconf_20190324_01.ds', 13, 1, [1,2,3,4]),
  Recording('S13-1_Seqconf_20190324_02.ds', 13, 1, [5,6,7,8]),
  Recording('S13-2_Seqconf_20190326_01.ds', 13, 2, [1,2,3,4]),
  Recording('S13-2_Seqconf_20190326_02.ds', 13, 2, [5,6,7,8]),  
  Recording('S14-1_Seqconf_20190323_01.ds', 14, 1, [1,2,3,4]),
  Recording('S14-1_Seqconf_20190323_02.ds', 14, 1, [5,6,7,8]),
  Recording('S14-2_Seqconf_20190324_01.ds', 14, 2, [1,2,3,4]),
  Recording('S14-2_Seqconf_20190324_02.ds', 14, 2, [5,6,7,8]),
  Recording('S15-1_Seqconf_20190327_01.ds', 15, 1, [1,2,3,4]),
  Recording('S15-1_Seqconf_20190327_02.ds', 15, 1, [5,6,7,8]),
  Recording('S15-2_Seqconf_20190328_01.ds', 15, 2, [1,2,3,4]),
  Recording('S15-2_Seqconf_20190328_02.ds', 15, 2, [5,6,7,8]),
  #Recording('Pilot01-01_Seqconf_20190123_01.ds', 2, 1, [1,2,3,4]), -> Run with coherences that were off
  Recording('Pilot01-01_Seqconf_20190123_02.ds', 2, 1, [5,6,7,8]),
  Recording('Pilot01-02_Seqconf_20190124_01.ds', 2, 2, [1,2,3,4]),
  Recording('Pilot01-02_Seqconf_20190124_02.ds', 2, 2, [5,6,7,8]),
  Recording('Pilot02-01_Seqconf_20190123_01.ds', 1, 1, [1,2,3,4]),
  Recording('Pilot02-01_Seqconf_20190123_03.ds', 1, 1, [5,6,7,8]),
  Recording('Pilot02-02_Seqconf_20190124_01.ds', 1, 2, [1,2,3,4]),
  Recording('Pilot02-02_Seqconf_20190124_02.ds', 1, 2, [5,6,7,8]),  
]


mapping = {20: ('response', 1),
           21: ['response', 2],
           22: ['response', 3],
           23: ['response', 4],
           29: ['response', 5],
           32: ['confidence_onset', 0],
           34: ['start_block', 0],
           33: ['end_block', 0],
           11: ['confidence', 1],
           12: ['confidence', 2],
           13: ['confidence', 3],
           14: ['confidence', 4],
           15: ['confidence', 5],
           16: ['confidence', 6],
           41: ['motion_on', 1],
           42: ['coherence_on', 1],
           43: ['irimotion', 1]}

def submit(recs=None):
    if recs is None:
        recs = np.arange(len(recordings))
    from pymeg import parallel
    for i in recs:                
          parallel.pmap(
              wrap_preprocess, [(i,)],
              walltime='15:00:00', memory=50, nodes=1, tasks=4,
              name='PREP_'+recordings[i].filename,
              ssh_to=None, env='py36')

def wrap_preprocess(i):
  return preprocess_raw(recordings[i])

def get_hash(subject, session, block, trial):
  '''
  2 sessions
  8 blocks
  120 trials per block
  960 per session
  = 1920 trials per subject
  '''
  return trial + 120*(block-1) + (120*8*(session-1)) + (1920*(subject-1))


#def get_blocks(raw):
#    #meta, timing = get_meta(raw, mapping, {}, 41, 41)
#    #meta.loc[:, 'hash'] = np.arange(len(meta))
#    #meta.loc[:, 'timing'] = np.arange(len(meta))
#
#    return blocks_from_marker(raw)
    

def get_preprocessed_block(raw, block):
    start, end = block.start*60, block.end*60 # To seconds
    r = raw.copy().crop(start, end)
    print('Processing ',r)
    block_meta, block_timing = get_meta(r, mapping, {}, 41, 41)

    # For subject 10 there is a spurious trigger in coherent_motion on 
    # Filter this one out
    try:
        if len(block_meta.loc[10, 'coherence_on']) == 2:
            block_meta.loc[10, 'coherence_on'] = 1
            block_timing.loc[10, 'coherence_on_time'] = block_timing.loc[10, 'coherence_on_time'][0]
    except TypeError:
        pass


    r, ants, artdef = preprocess_block(r)
    return r, ants, artdef, block_meta, block_timing

def blocks_from_marker(subject, raw, sfreq=1200.):
    '''
    Find breaks to cut data into pieces.

    Returns a dictionary ob blocks with a namedtuple that
    defines start and end of a block. Start and endpoint
    are inclusive (!).    
    '''
    events, _ = get_events(raw)
    if subject <= 2:
        #no triggers for start/endblock, so manually provide onset/end times
        trials = events[events[:,2]==42,0] #find all coh motion onsets
        assert(len(trials)==480)
        onsets = trials[[0, 120, 240, 360]]  / sfreq / 60 #manually select start & endpoint
        ends = trials[[119, 239, 359, 480-1]]  / sfreq / 60
    else:   
        onsets = events[events[:,2]==34, 0] / sfreq / 60
        ends = events[events[:,2]==33, 0] / sfreq / 60    

    Block = namedtuple('Block', ['start', 'end'])
    blocks = {block: Block(start, end) for block, (start, end)
              in enumerate(zip(onsets, ends))}
    return blocks

def filenames(subject, epoch, session, block):
    from os.path import join
    path = join(savepath, 'seqconf')
    fname = 'S%02i_%s_SESS%i_B%i' % (subject, epoch, session, block)
    return (join(path, fname + '-epo.fif.gz'),
            join(path, fname + '.meta'),
            join(path, fname + '.artdef'))


def preprocess_raw(recording):
    raw = mne.io.ctf.read_raw_ctf(
        join(inpath, recording.filename))
    blocks = blocks_from_marker(recording.subject, raw)
    min_start, max_end = np.min(raw.times), np.max(raw.times)

    for i, block in blocks.items():
        # Cut into blocks        
        r, ants, artdef, block_meta, block_timing = get_preprocessed_block(raw, block)
        print('Notch filtering')
        midx = np.where([x.startswith('M') for x in r.ch_names])[0]
        r.load_data()
        r.notch_filter(np.arange(50, 251, 50), picks=midx)        

        # Get unique trial indices
        index = get_hash(recording.subject, 
          recording.session, 
          recording.block[i], 
          np.arange(1, len(block_meta)+1))
        block_meta.loc[:, 'trial'] = index
        
        def uniquify(x):
          data = []
          for i in x.values:
            try:
              data.append(i[0])
            except (IndexError,TypeError,) as e:
              data.append(i)
          return data

        # Sometimes subjects press twice, keep only first event
        block_meta.loc[:, 'response'] = uniquify(block_meta.loc[:, 'response'])
        block_timing.loc[:, 'response_time'] = uniquify(block_timing.loc[:, 'response_time'])
        block_meta.loc[:, 'confidence'] = uniquify(block_meta.loc[:, 'confidence'])
        block_timing.loc[:, 'confidence_time'] = uniquify(block_timing.loc[:, 'confidence_time'])
        # Cut into epochs
        for epoch, event, (tmin, tmax), (rmin, rmax) in zip(
                ['stimulus', 'response', 'confidence'],
                ['coherence_on_time', 'response_time', 'confidence_time'],
                [(-1, 2.5), (-1.5, 1.5), (-1.5, 1.5)],
                [(-0.5, 1.5), (-.75, 0.75), (-1, 0.5)]):

            m, s = get_epoch(r, block_meta, block_timing,
                             event=event, epoch_time=(
                                 tmin, tmax),
                             reject_time=(rmin, rmax),
                             epoch_label='trial'
                             )
            artdef['drop_log'] = s.drop_log
            artdef['drop_log_stats'] = s.drop_log_stats()
            if len(s) <= 0:
                continue
            epofname, mfname, afname = filenames(recording.subject, 
              epoch, recording.session, recording.block[i])
            s = s.resample(600, npad='auto')
            s.save(epofname)
            m.to_hdf(mfname, 'meta')
            pickle.dump(artdef, open(afname, 'wb'), protocol=2)


def prepare_trans_fifs(move_to=None, recs=None):
    from pymeg import source_reconstruction as sr
    import os
    from pathlib import Path
    if recs is None:
      recs = recordings
    for recording in recs:
        try:
            rawname =  join(inpath, recording.filename)
            epoch_name, _, _ = filenames(recording.subject, 
                'stimulus', recording.session, recording.block[1])
            
            save_file = Path(sr.get_trans_epoch(rawname, epoch_name))
            print(rawname, epoch_name, '->', save_file)
            if move_to is not None:
                dest = Path(move_to) / save_file.name
                print('Copy to:', dest)
                os.rename(save_file, dest)
        except:
            print('Skipping', recording)


