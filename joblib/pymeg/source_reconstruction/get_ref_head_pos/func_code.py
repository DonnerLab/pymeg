# first line: 278
@memory.cache
def get_ref_head_pos(filename,  trans, N=-1):
    """Compute average head position from epochs.

    Args:
        filename: str
            Epochs file to load
        trans: dict
            A dictionary that contains t_ctf_dev_dev
            transformation matrix, e.g. output of
            get_ctf_trans
    Returns:
        Dictionary that contains average fiducial positions.
    """
    from mne.transforms import apply_trans
    data = preprocessing.load_epochs([filename])[0]
    cc = head_loc(data.decimate(10))
    nasion = np.stack([c[0] for c in cc[:N]]).mean(0)
    lpa = np.stack([c[1] for c in cc[:N]]).mean(0)
    rpa = np.stack([c[2] for c in cc[:N]]).mean(0)
    nasion, lpa, rpa = nasion.mean(-1), lpa.mean(-1), rpa.mean(-1)

    return {'nasion': apply_trans(trans['t_ctf_dev_dev'], np.array(nasion)),
            'lpa': apply_trans(trans['t_ctf_dev_dev'], np.array(lpa)),
            'rpa': apply_trans(trans['t_ctf_dev_dev'], np.array(rpa))}
