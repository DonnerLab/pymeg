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
import pandas as pd
from meg.tools import hilbert
from meg import artifacts
import logging
from joblib import Memory


memory = Memory(cachedir='/tmp/')


def get_trial_periods(events, trial_start, trial_end):
    '''
    Parse trial start and end times from events.
    '''
    start = [0,0,]
    end = [0,]
    if not len(start) == len(end):
        dif = len(start)-len(end)
        start = where(events[:,2] == trial_start)[0]
        end = where(events[:, 2] == trial_end)[0]

        # Aborted block during a trial, find location where [start ... start end] occurs
        i_start, i_end = 0, 0   # i_start points to the beginning of the current
                                # trial and i_end to the beginning of the current trial.

        if not (len(start) == len(end)):
            # Handle this condition by looking for the closest start to each end.
            id_keep = (0*events[:,0]).astype(bool)
            start_times = events[start, 0]
            end_times = events[end, 0]

            for i, e in enumerate(end_times):
                d = start_times-e
                d[d>0] = -inf
                matching_start = argmax(d)
                evstart = start[matching_start]

                if (trial_end in events[evstart-10:evstart, 2]):
                    prev_end = 10-where(events[evstart-10:evstart, 2]==trial_end)[0][0]
                    id_keep[(start[matching_start]-prev_end+1):end[i]+1] = True
                else:
                    id_keep[(start[matching_start]-10):end[i]+1] = True
            events = events[id_keep,:]

        start = where(events[:,2] == trial_start)[0]
        end = where(events[:, 2] == trial_end)[0]
    return start, end


def get_meta(raw, mapping, pins):
    '''
    Parse block structure from events in MEG files.

    Aggresively tries to fix introduced by recording crashes and late recording
    starts.

    '''

    def pins2num(pins):
        if len(pins) == 0:
            trial = 1
        else:
            # Convert pins to numbers
            trial = sum([2**(8-pin) for pin in pins])
        return trial

    events, _ = get_events_from_file(raw.info['filename'])
    events = events.astype(float)
    start, end = get_trial_periods(events, trial_start, trial_end)

    trials = []
    for i, (ts, te) in enumerate(zip(start, end)):
        current_trial = {}
        trial_nums = events[ts:ts, 2]
        trial_times = events[ts:ts, 0]
        if pins:
            # Find any pins that need special treatment, parse them and remove
            # triggers from trial_nums
            for key, value in pins.iteritems():
                if key in trial_nums:
                    pstart = where(trial_nums==key)[0]
                    pend = pstart + where(trial_nums[pstart:]>8)[0]
                    pvals = trial_nums[pstart:pend]
                    current_trial[value] = pins2num(pvals)
                    trial_nums = trial_nums[:pstart][pend-pstart,:]
                    trial_times = trial_times[:pstart][pend-pstart,:]

        for trigger, time in zip(trial_nums, trial_times):
            if trigger in mapping.keys():
                current_trial[mapping[trigger][0]] = mapping[trigger][1]
                current_trial[mapping[trigger][0]+'_time'] = time
            else:
                current_trial[trigger] = time
        trials.append(current_trial)
    return trials


def preprocess_block(raw):
    '''
    Apply artifact detection to a block of data.
    '''
    ab = artifacts.annotate_blinks(raw)
    am, zm = artifacts.annotate_muscle(raw)
    ac, zc, d = artifacts.annotate_cars(raw)
    ar, zj = artifacts.annotate_jumps(raw)
    ants = artifacts.combine_annotations([x for x in  [ab, am, ac, ar] if x is not None])
    ants.onset += raw.first_samp/raw.info['sfreq']
    raw.annotations = ants
    artdef = {'muscle':zm, 'cars':(zc, d), 'jumps':zj}
    return raw, ants, artdef


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
    joined_meta = pd.concat([meta, timing], axis=1)
    ev = metadata.mne_events(joined_meta, event, epoch_label)
    eb = metadata.mne_events(joined_meta, base_event, epoch_label)
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
    meta = meta.reset_index().set_index('hash').loc[sei]
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
        print t.stim_onset_t.min()
    timings = pd.concat(timings)
    metas = pd.concat(metas)
    return raw, metas, timings


def apply_baseline(epochs, baseline):
    drop_list = []
    for epoch, orig in enumerate(epochs.selection):
        # Find baseline epoch for this.
        base = where(baseline.selection==orig)[0]
        if len(base) == 0:
            # Reject this one.
            drop_list.append(epoch)
        else:
            base_val = squeeze(baseline._data[base, :, :]).mean(1)
            epochs._data[epoch, :, :] -= base_val[:, newaxis]

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