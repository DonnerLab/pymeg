# first line: 227
@memory.cache
def get_head_correct_info(raw_filename, epoch_filename, N=-1):
    """Get transformation matrix, fiducial positions and infor structure.

    The returned info structure contains fiducial locations computed from 
    the epoch data. 
    """
    trans = get_ctf_trans(raw_filename)
    fiducials = get_ref_head_pos(epoch_filename, trans, N=N)
    raw = mne.io.ctf.read_raw_ctf(raw_filename)
    info = replace_fiducials(raw.info, fiducials)
    return trans, fiducials, info
