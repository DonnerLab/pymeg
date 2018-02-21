'''
Preprocess an MEG data set.

The idea for preprocessing MEG data is modelled around a few aspects of the
confidence data set:
    1. Each MEG dataset is accompanied by a DataFrame that contains metadata for
       each trial.
    2. Trial meta data can be matched to the MEG data by appropriate triggers.
    3. Each MEG datafile contains several blocks of data that can be processed
       independently.

This leads to the following design:
    1. MEG is cut into recording blocks and artifact detection is carried out
    2. Each processed block is matched to meta data and timing (event on- and offsets)
       data is extracted from the MEG and aligned with the behavioral data.
    3. Data is epoched. Sync with meta data is guaranteed by a unique key for
       each trial that is stored along with the epoched data.
'''

import mne
import numpy as np
import pandas as pd
from pymeg.tools import hilbert
from pymeg import artifacts
import logging
from joblib import Memory


memory = Memory(cachedir='/tmp/')


def get_trial_periods(events, trial_start, trial_end):
    '''
    Parse trial start and end times from events.
    '''
    start = np.where(events[:,2] == trial_start)[0]
    end = np.where(events[:, 2] == trial_end)[0]
    if not len(start) == len(end):
        start_times = events[start, 0]
        start = []
        end_times = events[end, 0]
        for i, e in enumerate(end_times):
            d = start_times-e
            d[d>0] = -np.inf
            start_index = np.where(events[:, 0]==start_times[np.argmax(d)])[0][0]
            start.append(start_index)
    return np.array(start), end


def get_meta(raw, mapping, trial_pins, trial_start, trial_end, other_pins=None):
    '''
    Parse block structure from events in MEG files.

    Aggresively tries to fix introduced by recording crashes and late recording
    starts.

    mapping =
    '''

    def pins2num(pins):
        if len(pins) == 0:
            trial = 1
        else:
            # Convert pins to numbers
            trial = sum([2**(8-pin) for pin in pins])
        return trial

    events, _ = get_events(raw)
    events = events.astype(float)
    if trial_start == trial_end:
        start = np.where(events[:,2] == trial_start)[0]
        end = np.where(events[:, 2] == trial_end)[0]
        start, end = start[:-1], end[1:]
    else:
        start, end = get_trial_periods(events, trial_start, trial_end)

    trials = []
    for i, (ts, te) in enumerate(zip(start, end)):
        current_trial = {}
        trial_nums = events[ts:te+1, 2]
        trial_times = events[ts:te+1, 0]
        if trial_pins:
            # Find any pins that need special treatment, parse them and remove
            # triggers from trial_nums
            for key, value in trial_pins.iteritems():
                if key in trial_nums:
                    pstart = np.where(trial_nums==key)[0][0]+1
                    pend = pstart + np.where(trial_nums[pstart:]>8)[0][0] + 1
                    pvals = trial_nums[pstart:pend]
                    current_trial[value] = pins2num(pvals)
                    trial_nums = np.concatenate((trial_nums[:pstart], trial_nums[pend:]))
                    trial_times = np.concatenate((trial_times[:pstart], trial_times[pend:]))

        for trigger, time in zip(trial_nums, trial_times):
            if trigger in mapping.keys():
                key = mapping[trigger][0]
                val = mapping[trigger][1]
            else:
                key = trigger
                val = time
            if key in current_trial.keys():
                try:
                    current_trial[key].append(current_trial[key][-1] + 1)
                    current_trial[key +'_time'].append(time)
                except AttributeError:
                    current_trial[str(key)] = [current_trial[key], current_trial[key]+1]
                    current_trial[str(key) +'_time'] = [current_trial[str(key) +'_time'], time]
            else:
                current_trial[key] = val
                current_trial[str(key) +'_time'] = time
        trials.append(current_trial)

    meta = pd.DataFrame(trials)

    # Find other pins that are not trial related
    if other_pins:
        nums = events[:, 2]
        for key, value in other_pins.iteritems():
            pstarts = np.where(nums==key)[0] + 1
            for pstart in pstarts:
                t = events[pstart, 0]
                pend = pstart + np.where(nums[pstart:]>8)[0][0] + 1
                pvals = nums[pstart:pend]
                idx = meta.trial_start_time > t
                meta.loc[idx, value] = pins2num(pvals)

    time_fields = [c for c in meta if str(c).endswith('_time')]
    meta_fields = [c for c in meta if not str(c).endswith('_time')]
    return meta.loc[:, meta_fields], meta.loc[:, time_fields]


def preprocess_block(raw, blinks=True):
    '''
    Apply artifact detection to a block of data.
    '''
    ab = None
    artdef = {}
    if blinks:
        ab = artifacts.annotate_blinks(raw)
        artdef['blinks'] = ab
    am, zm = artifacts.annotate_muscle(raw)
    artdef['muscle'] = zm
    ac, zc, d = artifacts.annotate_cars(raw)
    artdef['cars'] = [zc, d]
    raw, ar, zj, jumps = artifacts.annotate_jumps(raw)
    artdef['jumps'] = zj
    ants = artifacts.combine_annotations([x for x in  [ab, am, ac, ar] if x is not None])
    #ants.onset += raw.first_samp/raw.info['sfreq']
    raw.annotations = ants
    artdef.update({'muscle':zm, 'cars':(zc, d), 'jumps':(zj, jumps)})
    return raw, ants, artdef


def mne_events(data, time_field, event_val):
    return np.vstack([
        data[time_field].values,
        0*data[time_field].values,
        data[event_val].values]).astype(int).T


def get_epoch(raw, meta, timing,
              event='stim_onset_t', epoch_time=(-.2, 1.5),
              base_event='stim_onset_t', base_time=(-.2, 0),
              epoch_label='hash'):
    '''
    Cut out epochs from raw data and apply baseline correction.

    Parameters
    ----------
    raw : raw data
    meta, timing : Dataframes that contain meta and timing information
    event : Column in timing that contains event onsets in sample time
    epoch_time : (start, end) in sec. relative to event onsets defined by 'event'
    base_event : Column in timing that contains baseline onsets in sample time
    base_time : (start, end) in sec. relative to baseline onset
    epoch_label : Column in meta that contains epoch labels.
    '''
    fields = set((event, base_event, epoch_label))
    all_meta = pd.concat([meta, timing], axis=1)
    joined_meta = (pd.concat([meta, timing], axis=1)
                    .loc[:, fields]
                    .dropna())

    ev = mne_events(joined_meta, event, epoch_label)
    eb = mne_events(joined_meta, base_event, epoch_label)

    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')

    base = mne.Epochs(raw, eb, tmin=base_time[0], tmax=base_time[1], baseline=None, picks=picks,
            reject_by_annotation=True)
    stim_period = mne.Epochs(raw, ev, tmin=epoch_time[0], tmax=epoch_time[1], baseline=None, picks=picks,
            reject_by_annotation=True)
    base.load_data()
    stim_period.load_data()
    stim_period, dl = apply_baseline(stim_period, base)
    # Now filter raw object to only those left.
    sei = stim_period.events[:, 2]
    meta = all_meta.reset_index().set_index(epoch_label).loc[sei]
    return meta, stim_period


def concat(raws, metas, timings):
    '''
    Concatenate a set of raw objects and apply offset to meta to
    keep everything in sync. Should allow to load all sessions of
    a subject. Can then crop to parallelize.
    '''
    raws = [r.copy() for r in raws]
    offsets = np.cumsum([0]+[len(raw) for raw in raws])
    raw = raws[::-1].pop()
    raw.append(raws, preload=False)
    timings = [timing+offset for timing, offset in zip(timings, offsets)]
    for t in timings:
        print(t.stim_onset_t.min())
    timings = pd.concat(timings)
    metas = pd.concat(metas)
    return raw, metas, timings


def apply_baseline(epochs, baseline):
    drop_list = []
    for epoch, orig in enumerate(epochs.selection):
        # Find baseline epoch for this.
        base = np.where(baseline.selection==orig)[0]
        if len(base) == 0:
            # Reject this one.
            drop_list.append(epoch)
        else:
            base_val = np.squeeze(baseline._data[base, :, :]).mean(1)
            epochs._data[epoch, :, :] -= base_val[:, np.newaxis]

    return epochs.drop(drop_list), drop_list


@memory.cache
def get_events_from_file(filename):
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    buttons = mne.find_events(raw, 'UPPT002', shortest_event=1)
    triggers = mne.find_events(raw, 'UPPT001', shortest_event=1)
    return triggers, buttons


def get_events(raw):
    buttons = mne.find_events(raw, 'UPPT002', shortest_event=1)
    triggers = mne.find_events(raw, 'UPPT001', shortest_event=1)
    return triggers, buttons


def load_epochs(filenames):
    return [mne.read_epochs(f) for f in filenames]


def load_meta(filenames):
    return [pd.read_hdf(f, 'meta') for f in filenames]


def concatenate_epochs(epochs, metas):
    '''
    Concatenate a list of epoch and meta objects and set their dev_head_t projection to
    that of the first epoch.
    '''
    dev_head_t = epochs[0].info['dev_head_t']
    index_cnt = 0
    epoch_arrays = []
    processed_metas = []
    for e, m in zip(epochs, metas):
        e.info['dev_head_t'] = dev_head_t
        processed_metas.append(m)
        e = mne.epochs.EpochsArray(e._data, e.info, events=e.events)
        epoch_arrays.append(e)
    return mne.concatenate_epochs(epoch_arrays), pd.concat(processed_metas)


def combine_annotations(annotations, first_samples, last_samples, sfreq):
    '''
    Concatenate a list of annotations objects such that annotations
    stay in sync with the output of mne.concatenate_raws.

    This function assumes that annotations objects come from different raw objects
    that are to be concatenated. In this case the concatenated raw object retains
    the first sample of the first raw object and then treats the data as
    continuous. In contrast, the annotation onsets already shifted by each individual
    raw object's first sample to be in sync. When concatenting annotations this
    needs to be taken into account.

    Parameters
    ----------
    annotations : list of annotations objects, shape (n_objects,)
    first_samples : list of ints, shape (n_objects,)
        First sample of each annotations' raw object.
    last_samples : list of ints, shape (n_objects,)
        Last sample of each annotations' raw object.
    sfreq : int
        Sampling frequency of data in raw objects.
    '''
    if all([ann is None for ann in annotations]):
        return None
    durations = [(1+l-f)/sfreq for f, l in zip(first_samples, last_samples)]
    offsets = np.cumsum([0] + durations[:-1])

    onsets = [(ann.onset-(fs/sfreq))+offset
                        for ann, fs, offset in zip(annotations, first_samples, offsets) if ann is not None]
    if len(onsets) == 0:
        return mne.annotations.Annotations(onset=[], duration=[], description=None)
    onsets = np.concatenate(onsets) + (first_samples[0]/sfreq)
    return mne.annotations.Annotations(onset=onsets,
        duration=np.concatenate([ann.duration for ann in annotations]),
        description=np.concatenate([ann.description for ann in annotations]))
