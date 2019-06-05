from pathlib import Path
import logging
import mne
import numpy as np
import os

from joblib import Memory

from os import makedirs
from os.path import join
from glob import glob

from pymeg import lcmv
from pymeg import preprocessing
from pymeg import source_reconstruction as sr
from pymeg.specific import prep_seqconf as ps

import pandas as pd
import pickle

from mne import compute_covariance

memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"])

# Epochs
epochs_path = Path("/home/nwilming/seqconf")
raw_path = Path("/home/kdesender/meg_data/seqconf/")
path = Path("/home/nwilming/seqconf/sr_labeled")
trans_path = Path("/home/nwilming/seqconf/trans")


def set_n_threads(n):
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def submit(recordings=None):
    if recordings is None:
        recordings = range(ps.recordings)
    from pymeg import parallel
    for recording in recordings:
        for epoch in ['stimulus', 'response']:
            for signal in ["HF", "LF"]:
                parallel.pmap(
                    extract,
                    [(recording, epoch, signal)],
                    walltime="15:00:00",
                    memory=50,
                    nodes=1,
                    tasks=4,
                    name="SR" + str(subject) + "_" + str(session) + str(recording),
                    ssh_to=None,
                    env="mne",
                )


def lcmvfilename(recording, signal, epoch, chunk=None):
    """
    Generate filename for where to save data.
    """
    try:
        makedirs(path)
    except:
        pass
    if chunk is None:
        filename = "S%s-SESS%i-B%i-%s-%s-lcmv.hdf" % (
            recording.subject,
            recording.session,
            recording.block[0],
            epoch,
            signal,
        )
    else:
        filename = "S%s-SESS%i-B%i-%s-%s-C%i-lcmv.hdf" % (
            recording.subject,
            recording.session,
            recording.block[0],
            epoch,
            signal,
            chunk,
        )
    return path / filename


def align_labels(subjects=np.arange(1, 16)):
    from pymeg import atlas_glasser as asr

    for subject in subjects:
        asr.get_hcp_annotation("/home/nwilming/seqconf/fsdir/", "SQC_S%02i" % subject)
        asr.get_JWDG_labels("SQC_S%02i" % subject, "/home/nwilming/seqconf/fsdir/")
        asr.get_JWDG_labels("SQC_S%02i" % subject, "/home/nwilming/seqconf/fsdir/")


def get_filenames(epoch, recording):
    """
    Return all epochs associated with recording and epoch
    """
    return [
        epochs_path / ps.filenames(recording.subject, epoch, recording.session, b)[0]
        for b in recording.block
    ]


@memory.cache
def get_cov(epochs, tmin=0, tmax=1):
    """Compute a covariance matrix with default settings.

    This is mainly a helper function to cache computation of covariance
    matrices.
    """
    return compute_covariance(epochs, tmin=tmin, tmax=tmax, method="auto")


@memory.cache()
def get_baseline(recording):
    filenames = get_filenames("stimulus", recording)
    print(filenames)
    epochs = preprocessing.load_epochs([str(f) for f in filenames])
    epochs = preprocessing.concatenate_epochs(epochs, None)
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith("M")])
    id_time = (-0.5 <= epochs.times) & (epochs.times <= -0.250)
    return epochs._data[:, :, id_time].mean(-1)[:, :, np.newaxis]


def get_epoch(epoch, recording):
    filenames = get_filenames(epoch, recording)
    epochs = preprocessing.load_epochs([str(f) for f in filenames])
    epochs = preprocessing.concatenate_epochs(epochs, None)
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith("M")])
    epochs._data -= get_baseline(recording)
    data_cov = get_cov(epochs, tmin=0, tmax=2)
    return data_cov, epochs


def extract(
    recording_number,
    epoch,
    signal_type="BB",
    BEM="three_layer",
    chunks=100,
    njobs=4,
    glasser_only=True,
):
    recording = ps.recordings[recording_number]
    mne.set_log_level("ERROR")
    lcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)

    logging.info("Reading stimulus data")
    data_cov, epochs = get_epoch(epoch, recording)
    raw_filename = raw_path / recording.filename
    trans_filename = trans_path / (
        "SQC_S%02i-SESS%i_B%i_trans.fif"
        % (recording.subject, recording.session, recording.block[1])
    )
    epoch_filename = ps.filenames(
        recording.subject, epoch, recording.session, recording.block[1]
    )[0]

    logging.info("Setting up source space and forward model")

    forward, bem, source = sr.get_leadfield(
        "SQC_S%02i" % recording.subject,
        str(raw_filename),
        str(epoch_filename),
        str(trans_filename),
        bem_sub_path="bem",
        sdir="/home/nwilming/seqconf/fsdir/",
    )
    if glasser_only:
        labels = sr.get_labels(
            "SQC_S%02i" % recording.subject,
            filters=["*wang2015atlas*"],
            sdir="/home/nwilming/seqconf/fsdir/",
        )
    else:
        labels = sr.get_labels(
            "SQC_S%02i" % recording.subject, sdir="/home/nwilming/seqconf/fsdir/"
        )
        labels = sr.labels_exclude(
            labels,
            exclude_filters=[
                "wang2015atlas.IPS4",
                "wang2015atlas.IPS5",
                "wang2015atlas.SPL",
                "JWDG_lat_Unknown",
            ],
        )
        labels = sr.labels_remove_overlap(labels, priority_filters=["wang", "JWDG"])

    fois_h = np.arange(36, 162, 4)
    fois_l = np.arange(2, 36, 1)
    tfr_params = {
        "HF": {
            "foi": fois_h,
            "cycles": fois_h * 0.25,
            "time_bandwidth": 2 + 1,
            "n_jobs": njobs,
            "est_val": fois_h,
            "est_key": "HF",
            "sf": 600,
            "decim": 10,
        },
        "LF": {
            "foi": fois_l,
            "cycles": fois_l * 0.4,
            "time_bandwidth": 1 + 1,
            "n_jobs": njobs,
            "est_val": fois_l,
            "est_key": "LF",
            "sf": 600,
            "decim": 10,
        },
    }

    events = epochs.events[:, 2]
    filters = lcmv.setup_filters(epochs.info, forward, data_cov, None, labels)
    set_n_threads(1)

    for i in range(0, len(events), chunks):
        filename = lcmvfilename(
            recording, signal_type, epoch, 
            chunk=i,
        )
        if os.path.isfile(filename):
            continue
        if signal_type == "BB":
            logging.info("Starting reconstruction of BB signal")
            M = lcmv.reconstruct_broadband(
                filters,
                epochs.info,
                epochs._data[i : i + chunks],
                events[i : i + chunks],
                epochs.times,
                njobs=1,
            )
        else:
            logging.info("Starting reconstruction of TFR signal")
            M = lcmv.reconstruct_tfr(
                filters,
                epochs.info,
                epochs._data[i : i + chunks],
                events[i : i + chunks],
                epochs.times,
                est_args=tfr_params[signal_type],
                njobs=4,
            )
        M.to_hdf(str(filename), "epochs")
    set_n_threads(njobs)


def submit_aggregates(subjects=range(1, 16), 
        epochs=['stimulus', 'response', 'confidence'], 
        cluster="uke", 
        only_glasser=True):
    from pymeg import parallel
    import time

    for subject, epoch, session in product(
        subjects, epochs, range(2)
    ):
        parallel.pmap(
            aggregate,
            [(subject, session, epoch, )],
            name="agg" + str(session) + epoch + str(subject),
            tasks=8,
            memory=60,
            walltime="12:00:00",
        )


def aggregate(subject, session, epoch):
    """
    Aggregate source recon files into easier to process HDF files.

    The aggregates concatenate blocks from one session and different frequency 
    sets into the same file.
    """
    
    from pymeg import aggregate_sr as asr, atlas_glasser as ag
    from os.path import join

    stim_files = path.glob('S%i-SESS%i-stimulus-*-chunk*-lcmv.hdf'%(subject, session))
    files = path.glob('S%i-SESS%i-%s-*-chunk*-lcmv.hdf'%(subject, session, epoch))

    all_clusters = get_glasser_clusters()

    filename = join(
        "/home/nwilming/conf_meg/sr_labeled/aggs/",
        "S%i_SESS%i_%s_agg.hdf" % (subject, session, epoch),
        )
    print("Will save agg as:", filename)    
    agg = asr.aggregate_files(files, stim_files, (-0.5, -0.25), all_clusters=all_clusters)
    asr.agg2hdf(agg, filename)


def get_glasser_clusters():
    """
    Define each Glasser area as a cluster.
    """
    from pymeg import atlas_glasser as ag
    # fmt: off
    areas = ["TE1a", "VVC", "FOP1", "10v", "6r", "H", "LIPv", "OFC", "PFop", "STSvp",
             "VMV1", "STGa", "p24pr", "TGv", "3a", "p9-46v", "9m", "IFJp", "LIPd", "pOFC",
             "IPS1", "7PC", "PIT", "V1", "SCEF", "i6-8", "25", "PoI2", "a24pr", "8Av",
             "V2", "p47r", "V4", "p10p", "10d", "3b", "a24", "TA2", "10pp", "AIP",
             "PCV", "6d", "TF", "31pd", "FOP5", "MST", "IP1", "LO3", "PH", "45",
             "8Ad", "s6-8", "VMV2", "a47r", "46", "a9-46v", "FOP2", "V3CD", "PEF", "ProS",
             "p24", "MI", "PreS", "STSdp", "a10p", "MBelt", "FFC", "VMV3", "V3A", "VIP",
             "PoI1", "TE2p", "52", "9a", "31pv", "PeEc", "PI", "IFSp", "4", "A4",
             "AAIC", "RI", "5m", "23c", "7m", "PGp", "PHT", "p32", "6a", "FOP4",
             "PGi", "47l", "PGs", "MT", "55b", "A1", "TPOJ2", "Pir", "PHA3", "LO2",
             "IFSa", "p32pr", "8C", "LO1", "d32", "V3B", "V3", "V7", "6mp", "AVI",
             "d23ab", "31a", "DVT", "47m", "PFt", "9-46d", "A5", "V4t", "TPOJ1", "1",
             "PF", "6v", "PBelt", "OP2-3", "PHA2", "V8", "V6", "STSva", "44", "v23ab",
             "7Pm", "24dd", "TE1p", "OP4", "TE1m", "2", "V6A", "8BM", "IFJa", "10r",
             "IP0", "43", "OP1", "TE2a", "7Am", "6ma", "PFcm", "47s", "TPOJ3", "33pr",
             "FEF", "STSda", "MIP", "23d", "13l", "PHA1", "Ig", "24dv", "11l", "a32pr",
             "FST", "s32", "STV", "5mv", "9p", "TGd", "RSC", "POS1", "PFm", "IP2",
             "EC", "POS2", "FOP3", "LBelt", "PSL", "SFL", "5L", "7AL", "7PL", "9m"]
    areas = ["7AL", "7PL", "9m", "8BL"]
    # fmt: on

    areas = {
        area: ["L_{}_ROI-lh".format(area), "R_{}_ROI-rh".format(area)]
        for area in areas
    }
    return areas
