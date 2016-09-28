import meg
import mne
import numpy as np


filename = '/home/gortega/megdata/SHo_Attractor_20160925_01.ds'
raw = mne.io.read_raw_ctf(filename)
mapping = {30: ('wait_fix', 0),
       40: ('baseline_start', 0),
       50: ('dot_onset', 0),
       60: ('decision_start', 0),
       61: ('response', -1),
       62: ('response', 1),
       70: ('feedback', 1),
       80: ('rest', 80)}

def get_meta(raw, mapping):
    meta, timing = meg.preprocessing.get_meta(raw, mapping, {}, 151, 151)
    for c in meta:
        if c in [v[0] for v in mapping.values()] or str(c).endswith('time'):
            continue
        del meta[c]

    meta.loc[:, 'hash'] = timing.decision_start_time.values
    meta.loc[:, 'block'] = meta.index.values/102
    #timing = timing.set_index('hash')
    timing.loc[:, 'block'] = meta.block
    return meta, timing

meta, timing = get_meta(raw, mapping)

for i, ((bnum, mb), (_, tb)) in enumerate(zip(meta.groupby('block'), timing.groupby('block'))):
    r = raw.copy()
    r.crop(tmin=(tb.baseline_start_time.min()/1200.)-10, tmax=10+(tb.feedback_time.max()/1200.))
    mb, tb = get_meta(r, mapping)
    r, ants, artdef = meg.preprocessing.preprocess_block(r, blinks=False)
    slmeta, stimlock = meg.preprocessing.get_epoch(r, mb, tb, event='decision_start_time', epoch_time=(-2.5, .5),
        base_event='baseline_start_time', base_time=(0, 0.5), epoch_label='hash')
    slmeta.to_hdf('sl_meta_%i_ds1.hdf'%i, 'meta')
    stimlock.save('sl_meta_%i_ds1-epo.fif.gz'%i)
    del slmeta
    del stimlock
    rlmeta, resplock = meg.preprocessing.get_epoch(r, mb, tb, event='response_time', epoch_time=(-2.5, .5),
        base_event='baseline_start_time', base_time=(0, 0.5), epoch_label='hash')
    rlmeta.to_hdf('resp_meta_%i_ds1.hdf'%i, 'meta')
    resplock.save('resp_meta_%i_ds1-epo.fif.gz'%i)
