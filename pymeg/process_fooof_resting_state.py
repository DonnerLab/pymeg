import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
from scipy.signal import periodogram

# MNE & associated code
import mne
from mne.preprocessing import ICA, read_ica
from mne.utils import _time_mask

# FOOOF, and custom helper & utility functions
from fooof import FOOOF, FOOOFGroup
from fooof.objs import average_fg
from fooof.plts import plot_spectrum
from fooof.utils import trim_spectrum
from fooof.data import FOOOFSettings
from fooof.analysis import get_band_peak_fm, get_band_peak_fg
from fooof.bands import Bands
from fooof.utils import interpolate_spectrum
from fooof.plts import plot_spectra

###### 
### https://github.com/TomDonoghue/EEGparam/blob/main/notebooks/01-ProcessEEG.ipynb
####

# Set whether to drop outlier subjects, in terms of FOOOF Goodness-of-Fit metrics
drop_outliers = False

# Set average function to use
avg_func = np.nanmean
#avg_func = np.nanmedian

# Set FOOOF peak params label
peak_label = 'peak_params' # 'peak_params', 'gaussian_params'

controls = np.arange(101,107).tolist() + np.arange(108, 128).tolist()
scz = np.arange(2,23).tolist()  + np.arange(24,32).tolist() 
    
def get_psds_results(subjects, fmin, fmax):
    psds_all = []
    for subj in subjects:
        epo = mne.read_epochs('sj' + str(subj) + '_resting_state_ica_FCz_ref-epo.fif')
        if not 'FCz' in epo.info.ch_names:
            epo = mne.add_reference_channels(epo, ref_channels=['FCz'])
        if not 'Fz' in epo.info.ch_names:
            epo = mne.add_reference_channels(epo, ref_channels=['Fz'])
        epo = epo.copy().set_eeg_reference('average')
        srate = epo.info['sfreq']
        # PSD settings
        n_fft, n_overlap, n_per_seg = int(2*srate), int(srate), int(2*srate)
        # Calculate PSDs (across all channels) - from the first 2 minute of data
        psds, freqs = mne.time_frequency.psd_welch(
            epo, fmin=fmin, fmax=fmax, 
            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, verbose=None)
        interp_range = [49, 51]
        channels = epo.ch_names
        psds_interp = np.zeros((len(channels), np.shape(psds)[2]))
        print('subj', subj)
        for ch in channels:
            ch_ind = epo.ch_names.index(ch)
            freqs_int1, powers_int1 = interpolate_spectrum(freqs, psds[0,ch_ind,:], interp_range)
            psds_interp[ch_ind, :] = powers_int1
        psds_all.append(psds_interp)
    return psds_all, freqs


def get_fooof(psds_all, freqs, f_range):
    fgs = []
    for psds in psds_all:
       fg.fit(freqs, psds, f_range, progress='tqdm.notebook')
       fgs.append(fg.copy())
    return fgs


# Data settings
fmin, fmax = (1, 90)
psds_ctrls, freqs = get_psds_results(controls, fmin, fmax)
psds_scz, freqs = get_psds_results(scz, fmin, fmax)

# Initialize FOOOFGroup object
f_range = [30,70]#[3, 90]#[3, 25]   [30,70]#
fg = FOOOFGroup(peak_width_limits=[1, 6], max_n_peaks=6,
                min_peak_height=0.05, peak_threshold=1.5)
# Compute fooof
fgs_controls = get_fooof(psds_ctrls, freqs, f_range)
fgs_scz = get_fooof(psds_scz, freqs, f_range)

# Extract aperiodic parameters for each subject
def get_exponents(fgs, ch):
    ch_ind = epo.ch_names.index(ch)
    aps = np.empty(shape=[len(fgs), 2])
    mean_exp = np.empty(shape=[len(fgs), 1])
    all_exps_all_sj = []
    for ind, fg in enumerate(fgs):
        if len(fg.get_params('aperiodic_params')) == 64:
   #         print(len(fg.get_params('aperiodic_params')))
   #         print(ind)
            all_exps = fg.get_params('aperiodic_params', 'exponent')
            all_exps_all_sj.append(all_exps)
        aps[ind, :] = fg.get_params('aperiodic_params')[ch_ind]
        mean_exp[ind] = np.nanmean(fg.get_params('aperiodic_params')[:,1]) #mean across channels
    offs = aps[:, 0]
    exps = aps[:, 1]
    return exps, mean_exp, all_exps_all_sj, aps

exps_controls, mean_controls, all_exps_controls, aps_controls = get_exponents(fgs_controls, 'Cz')
exps_scz, mean_scz, all_exps_scz, aps_scz = get_exponents(fgs_scz, 'Cz')

ch_ind = epo.ch_names.index('Cz')
exp_cz_ctrls = []
exp_cz_scz = []
for i in range(len(all_exps_controls)):
    exp_cz_ctrls.append(all_exps_controls[i][ch_ind])
for i in range(len(all_exps_scz)):
    exp_cz_scz.append(all_exps_scz[i][ch_ind])

t, p = scipy.stats.ttest_ind(exp_cz_scz, exp_cz_ctrls, equal_var=False)
t, p = scipy.stats.ttest_ind(exps_scz, exps_controls, equal_var=False)
t_mean_chs, p_mean_chs = scipy.stats.ttest_ind(mean_scz, mean_controls, equal_var=False)

# Compute stats for each channel and print significant channels
for ch in epo.info['ch_names']:
    ch_ind = epo.ch_names.index(ch)
    exp_ch_ctrls = []
    exp_ch_scz = []
    for i in range(len(all_exps_controls)):
        exp_ch_ctrls.append(all_exps_controls[i][ch_ind])
    for i in range(len(all_exps_scz)):
        exp_ch_scz.append(all_exps_scz[i][ch_ind])
    t, p = scipy.stats.ttest_ind(exp_ch_scz, exp_ch_ctrls, equal_var=False)
    if p < 0.05:
        print(ch, p)

# Plot Topoplot
mne.viz.plot_topomap(np.nanmean(all_exps_controls, axis = 0), epo.info, cmap=cm.viridis, contours=0);
mne.viz.plot_topomap(np.nanmean(all_exps_scz, axis = 0), epo.info, cmap=cm.viridis, contours=0);
plt.show()

# Set channel of interest 
ch_ind_oz = epo.ch_names.index('Oz')
ch_ind_cz = epo.ch_names.index('Cz')
srate = epo.info['sfreq']
eeg_data = epo
 
ch_ind = epo.ch_names.index('CP5')
# Plot a segment of data - to eyeball
start = 0 # Index to start plotting at, in samples
inds = [start, start + 10*srate]  # 10 seconds of data
fig = plt.figure(figsize=[16, 6])
plt.plot(eeg_data.times[0:10000],eeg_data._data[0,ch_ind, 0:10000])
#plt.plot(eeg_data.times,eeg_data._data[0,ch_ind, :])
plt.show()

# Get individual power spectrum of interest
for i in range(len(psds_scz)):
    psds = psds_scz[i]
    cur_psd = psds[ch_ind, :]
    # Get the peak within the alpha range
    al_freqs, al_psd = trim_spectrum(freqs, cur_psd, [7, 14])
    icf_ind = np.argmax(al_psd)
    al_icf = al_freqs[icf_ind]
    # Plot the power spectrum, with the individually detected alpha peak
    plot_spectrum(freqs, cur_psd, log_powers=True, ax=plt.subplots(figsize=(5, 5))[1])
    plt.plot(al_icf, np.log10(al_psd[icf_ind]), '.', markersize=12)
    # interp_range = [49, 51]
    # freqs_int1, powers_int1 = interpolate_spectrum(freqs, cur_psd, interp_range)

    # plot_spectra(freqs, [cur_psd, powers_int1], log_powers=True,
    #          labels=['Original Spectrum', 'Interpolated Spectrum'])

    # Run FOOOF across all power spectra
    f_range=[30,70]
    fg.fit(freqs, psds, f_range, progress='tqdm.notebook')

    # Check FOOOF model fit of particular channel of interest
    fm = fg.get_fooof(ch_ind, True)
    fm.print_results()
    fm.plot()
    plt.show()


BANDS = Bands({'alpha' : [7, 14]})
# Extract individualized CF from channel Oz
#fooof_freq, _, fooof_bw = get_band_peak_fm(fm, BANDS.alpha)

# Extract alphas from each subject
def get_alpha(fgs):
    alphas = np.empty(shape=[len(fgs), 3])
    for ind, fg in enumerate(fgs):
        alphas[ind, :] = get_band_peak_fg(fg, BANDS.alpha, attribute=peak_label)[ch_ind_oz, :]
    return alphas

alphas_scz = get_alpha(fgs_scz)
alphas_ctrls = get_alpha(fgs_controls)
# # If not FOOOF alpha extracted, reset to 10
# if np.isnan(fooof_freq):
#     fooof_freq = 10

# # Compare peak-find alpha peak to fooof alpha peak
# print('IndiPeak CF: \t{:0.2f}'.format(al_icf))
# print('FOOOF    CF: \t{:0.2f}'.format(fooof_freq))

# # Check extracted FOOOF alpha properties
# print('Alpha CF: \t{:0.2f}'.format(fooof_freq))
# print('Alpha BW: \t{:0.2f}'.format(fooof_bw))



# Check a summary of the FOOOFGroup results across all channels
#fg.plot()



# Extract some FOOOF data across all electrodes

# Extract exponents - all channels
all_exps = fg.get_params('aperiodic_params', 'exponent')

# Extract alpha oscillations - all channels
alphas = get_band_peak_fg(fg, BANDS.alpha)

# Extract aperiodic parameters for each subject
aps = np.empty(shape=[len(fgs), 2])
for ind, fg in enumerate(fgs):
    aps[ind, :] = fg.get_params('aperiodic_params')[ch_ind_cz]
offs = aps[:, 0]
exps = aps[:, 1]


# Extract error & R^2 from FOOOF fits
def extract_errs(fgs):
    errs = []; r2s = []
    for fg in fgs:
        errs.append(fg.get_results()[ch_ind_oz].error)
        r2s.append(fg.get_results()[ch_ind_oz].r_squared)

    errs = np.array(errs)
    r2s = np.array(r2s)
    return errs, r2s


errs_scz, r2s_scz = extract_errs(fgs_scz)
errs_ctrls, r2s_ctrls = extract_errs(fgs_controls)
# Settings for outlier check
std_thresh = 2.5

def check_outliers(data, thresh):
    """Calculate indices of outliers, as defined by a standard deviation threshold."""

    return list(np.where(np.abs(data - np.mean(data)) > thresh * np.std(data))[0])

# Check for outliers using FOOOF Goodness of Fit metrics
err_outliers_scz = check_outliers(errs_scz, std_thresh)
std_outliers_scz = check_outliers(r2s_scz, std_thresh)
err_outliers_ctrls = check_outliers(errs_ctrls, std_thresh)
std_outliers_ctrls = check_outliers(r2s_ctrls, std_thresh)

# Check if the same outliers are detected for each GoF measure
print(err_outliers_scz)
print(std_outliers_scz)
print(err_outliers_ctrls)
print(std_outliers_ctrls)
# Combine outlier list across GoF metrics
bad_inds = set(err_outliers + std_outliers)


# Plot alpha topography
data_ind = 1 # 0:CF; 1:PW; 2:BW

# For sake of visualization, replace any NaN with the mean
temp_data = alphas[:, data_ind]
inds = np.where(np.isnan(temp_data))
temp_data[inds] = np.nanmean(temp_data)

mne.viz.plot_topomap(temp_data, eeg_data.info, cmap=cm.viridis, contours=0);
# Plot exponent topography
mne.viz.plot_topomap(all_exps, eeg_data.info, cmap=cm.viridis, contours=0);


# Filter data to canonical alpha band: 8-12 Hz
alpha_data = eeg_data.copy()
alpha_data.filter(8, 12, fir_design='firwin')



# Filter data to FOOOF derived alpha band
fooof_data = eeg_data.copy()
fooof_data.filter(fooof_freq-2, fooof_freq+2, fir_design='firwin')



# Plot the differently filtered traces - check for differences
inds = [200000, 201000] # Arbitrary time points
fig = plt.figure(figsize=[16, 6])
plt.plot(alpha_data.times[inds[0]:inds[1]],
         alpha_data._data[0,ch_ind, inds[0]:inds[1]],
         'b', label='Canonical')
plt.plot(fooof_data.times[inds[0]:inds[1]],
         fooof_data._data[0,ch_ind, inds[0]:inds[1]],
         'r', label='FOOOFed')
plt.legend()
plt.show()
# Take the analytic amplitude (hilbert) of the alpha-filtered signals
alpha_data.apply_hilbert(envelope=True)
fooof_data.apply_hilbert(envelope=True)
    

