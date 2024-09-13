'''
Tools for EEG processing
Author: Timofei Ponomarev
ORCID: 0009-0005-0308-8006

'''

import numpy as np
import os
import mne
from mne.decoding import Scaler


def signaltonoise(a:np.ndarray, axis=0, ddof=0):
    '''Calculate signal to noise ratio'''
    
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m**2/sd**2)
    # return m**2/sd**2

def SNR_peak(arr, target_interval, ref_interval, polarity):
    '''Calculate signal to noise ratio based on peak amplitude
    
    SNR of peak amplitude compared to baseline
    
    '''
    
    frag = arr[target_interval[0]:target_interval[1]]
    amp, _ = get_peak_amplitude(frag, np.arange(len(arr)), polarity)
    return np.abs(amp)/np.std(arr[ref_interval[0]:ref_interval[1]], axis=0)

def check_path(path):
    '''Check path and create necessary folders if absent'''
    
    if not os.path.exists(path):
        os.makedirs(path)
        
from scipy.signal import argrelmax, argrelmin

def get_peak_amplitude(x:np.ndarray, times:np.ndarray, polarity='pos'):
    '''Get local min/max for given array
    
    x: 1d array of eeg signal len of times
    times: corresponding array of times
    polarity: ['pos']itive or ['neg']ative
    
    Returns:
    amp - peak amplitude of a given signal
    lat - latency of the peak
    
    '''
    
    # Organize functions
    extremum = {
        'pos': argrelmax,
        'neg': argrelmin
    }
    absolute = {
        'pos': np.argmax,
        'neg': np.argmin
    }
    # Try to find local extremum
    id_extr = extremum[polarity](x)[0]
    # If no extremum, pick absolute value
    if id_extr.shape[0] == 0:
        id_extr = absolute[polarity](x)
    elif id_extr.shape[0] == 1:
        id_extr = id_extr[0]
    # Extract value
    amp = x[id_extr]
    lat = times[id_extr]
    # Chose one from many
    if amp.size>1:
        id = absolute[polarity](amp)
        amp = amp[id]
        lat = lat[id]
    
    return amp, lat

def make_epochs_array(data:np.ndarray, epochs:mne.Epochs, **kwargs):
    '''Create a mne.Epochs instance from any array data
    
    Set up and run mne.EpochsArray function.
    
    Parameters
    ----------
    * data -- an array
    * epochs -- mne.Epochs instance to exctract info, tmin, event_id, events and baseline
    
    '''
    info = kwargs['info'] if 'info' in kwargs.keys() else epochs.info
    tmin = kwargs['tmin'] if 'tmin' in kwargs.keys() else epochs.tmin
    event_id = kwargs['event_id'] if 'event_id' in kwargs.keys() else epochs.event_id
    events = kwargs['events'] if 'events' in kwargs.keys() else epochs.events
    baseline = kwargs['baseline'] if 'baseline' in kwargs.keys() else epochs.baseline
    
    return mne.EpochsArray(data, info, events, tmin,
                           event_id, baseline=baseline)

def scaler(epochs, separate_channels=False, adjust_scale=1):
    '''Scale the mne.Epochs data
    
    Parameters
    ----------
    
    * epochs - mne.Epochs
    * separate_channels - bool, whether to apply normalization
                          to each channel separately
    * adjust_scale - coefficient to adjust scale after normalization
                     (1e-6 to return the scale to microvolts e.g.)
    
    '''
    
    baseline_data = epochs.copy().crop(-0.2, 0).get_data(copy=True)
    ids = np.random.choice(np.arange(baseline_data.shape[0]), 45, False)
    if separate_channels:
        std = baseline_data[ids].std(axis=(0,2))
        data = epochs.get_data(copy=True).transpose(0, 2, 1)
        normdata = (data/std).transpose(0, 2, 1)
    else:
        vector = baseline_data[ids].mean(1).ravel()
        std = vector.std()
        data = epochs.get_data(copy=True)
        normdata = data/std
        
    return make_epochs_array(normdata*adjust_scale, epochs), std

def scaler_mne(epochs, adjust_scale=1):
    '''Apply scaler from the MNE package
    
    Parameters
    ----------
    
    * epochs - mne.Epochs
    * adjust_scale - coefficient to adjust scale after normalization
                     (1e-6 to return the scale to microvolts e.g.)
    
    '''
    
    scl = Scaler(epochs.info, scalings='mean')
    Xt = scl.fit_transform(epochs.get_data(copy=True))*adjust_scale
    
    return make_epochs_array(Xt, epochs)

def remove_evoked_component(epochs:mne.Epochs):
    '''Remove evoked component from Epochs'''
    
    adjusted_epochs = dict.fromkeys(epochs.event_id.keys())
    epochs.load_data()
    for ev in epochs.event_id.keys():
        epoch_ev = epochs[ev].copy().pick('eeg')
        ep = epoch_ev.get_data(copy=True)
        ave = epochs[ev].average().get_data()
        adj = ep - ave
        adjusted_epochs[ev] = mne.EpochsArray(adj, epoch_ev.info, event_id={ev:1}, tmin=-0.2)
    return adjusted_epochs