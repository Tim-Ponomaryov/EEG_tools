'''
Tools for EEG spatial filtering
Author: Timofei Ponomarev
ORCID: 0009-0005-0308-8006

'''

import numpy as np
import mne
from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm

from scipy.linalg import eig


def FCsf(X:np.ndarray, Y:np.ndarray):
    '''
    Fisher's criterion for spatial filtering (dimensionality reduction)

    X: EEG data of class 1 (channel x sample x trial)
    Y: EEG data of class 2 (channel x sample x trial)
    W: the colunms of projection matrix W are the spatial filters
    d: eigenvalues
    A: spatial patterns
    theta: regularization parameter, 1-FC, 0-CSP

    by Yu Zhang, ECUST & RIKEN, June 2012. Email: yuzhang@ecust.edu.cn
    Python conversion by Timofei Ponomarev & Anatoly Vasilyev, June 2024. Email: timofeyponomaryov@gmail.com
    
    '''
    
    # Compute spatial between-class matrix and spatial within-class matrix
    
    p_X = X.shape[2] / (X.shape[2]+X.shape[2])
    p_Y = Y.shape[2] / (Y.shape[2]+Y.shape[2])
    X_mean = np.mean(X, axis=2)
    Y_mean = np.mean(Y, axis=2)
    all_mean = (X_mean + Y_mean)/2

    S_b = np.matmul((X_mean - Y_mean),(X_mean - Y_mean).T)
    S_b = S_b/X_mean.shape[1]

    S_w1 = np.zeros((X.shape[0], X.shape[0]))
    S_w2 = np.zeros((Y.shape[0], Y.shape[0]))

    for i in range(0, X.shape[2]):
        S_w1 = S_w1 + np.matmul((X[:,:,i] - X_mean),(X[:,:,i] - X_mean).T)
        
    for i in range(0, Y.shape[2]):
        S_w2 = S_w2 + np.matmul((Y[:,:,i] - Y_mean),(Y[:,:,i] - Y_mean).T)
        
    S_w = S_w1 / np.prod(np.array([X.shape[1], X.shape[2]])) + \
            S_w2 / np.prod(np.array([Y.shape[1], Y.shape[2]]))

    # Solve FC spatial filters
            
    D, W = eig(S_b, S_w, left=True, right=False)
    # D = np.diag(D)
    A = np.linalg.pinv(W).T

    d, p = A.shape
    maxind = np.argmax(np.abs(A), axis=0)
    rowsign = np.sign(A.ravel('F')[maxind + np.arange(0,(d)*p,p)])
    W = W * rowsign
    A = np.linalg.pinv(W).T

    d = np.sort(D)[::-1]
    idx = np.argsort(D)[::-1]

    W = W[:, idx].T # transpose because of python row-major indexing
    A = A[:, idx].T
        
    return W, d, A  
        
def permute_FC(X:np.ndarray, Y:np.ndarray, nperm=500):
    '''Permutation test for validation of spatial filters'''
    
    D = [None] * nperm
    s_all = np.concatenate((X, Y), axis=2)
    labels = np.concatenate((np.ones(X.shape[2]), np.zeros(Y.shape[2])))

    for pp in tqdm(range(nperm)):
        cur_labels = np.random.permutation(labels)
        _, D[pp], _ = FCsf(s_all[:, :, cur_labels == 1], s_all[:, :, cur_labels == 0])

    return D

def calculate_sf(epochs:mne.Epochs, tmin=None, tmax=None,
                 equalize_epoch_counts=False):
    '''Pipeline to calculate spatial filters
    
    epochs must be a mne.Epochs instance with 2 event classes
    e.g. critical yes or no
    
    Returns
    -------
    * W - spatial filters
    * d - eigenvalues,
    * A - sparital projectioins
        
    
    '''
    # Alter epochs
    ep = epochs.copy().crop(tmin, tmax)
    evidn, evidy = ep.event_id.keys()
    ep1, ep2 = ep[evidn], ep[evidy]
    if equalize_epoch_counts:
        mne.epochs.equalize_epoch_counts([ep1, ep2])
    # Prepare arrays
    X, Y = ep1.get_data(copy=True), ep2.get_data(copy=True)
    X = np.transpose(X, (1, 2, 0)) # (ch x time x epochs)
    Y = np.transpose(Y, (1, 2, 0)) # (ch x time x epochs)
    # Calculater spatial filters
    W, d, A = FCsf(X, Y)
    
    return W, d, A

def apply_sf(epochs:mne.Epochs, W:np.ndarray, tmin=-0.2, baseline=(None,0)):
    '''Apply spatial filter to epochs
    
    Parameters
    ----------
    * epochs: mne.Epochs instance
    * W: spatial filter which is vector len of n_channels
    
    Returns
    -------
    Adjusted epochs
    
    '''
    
    X = epochs.get_data(copy=True)
    X = np.transpose(X,(0,2,1)) # Ch to last dim
    # W = W[0]
    Xf = X @ W
    info = mne.create_info(ch_names=['ch'], ch_types=['eeg'], sfreq=1000)
    
    return mne.EpochsArray(np.expand_dims(Xf,1), info,
                           event_id=epochs.event_id, events=epochs.events,
                           tmin=tmin, baseline=baseline)
    
def inverse_transform(epochs, W, A, drop_id=[0]):
    '''Perform the inverse transformation for sf'''
    
    # Prepare epochs
    X = epochs.get_data(copy=True)
    X = X.transpose((0, 2, 1)) # epochs x times x channels
    # Prepare matrices
    W1 = np.delete(W, drop_id, 0)
    A1 = np.delete(A, drop_id, 0)
    # Inverse transform
    Xt = (X @ W1.T) @ A1
    Xt = Xt.transpose(0, 2, 1) # epochs x channels x times
    
    return mne.EpochsArray(Xt, epochs.info, epochs.events, event_id=epochs.event_id,
                           tmin=epochs.tmin, baseline=epochs.baseline)
    
def plot_component(epochs:mne.Epochs, W:np.ndarray, A:np.ndarray, component_n):

    if isinstance(component_n, int):
        _plot_component(epochs, W, A, component_n)
    else: 
        for n in component_n:
            _plot_component(epochs, W, A, n)
    
def _plot_component(epochs:mne.Epochs, W:np.ndarray, A:np.ndarray, component_n:int):
    '''
    Plot spatial pattern and the waveform of componen
    
    '''
    # Apply component
    component = apply_sf(epochs, W[component_n])
    sp = A[component_n]
    # Create figure
    fig, axes = plt.subplots(1,2, width_ratios=(1,2), figsize=(9,3))
    ax_sp, ax_component = axes
    # Plot spatial pattern
    sp = mne.EvokedArray(np.expand_dims(sp, 1), epochs.info)
    sp.plot_topomap(times=0, colorbar=False, cmap='jet', axes=ax_sp, show=False)
    ax_sp.set_title('Spatial pattern')
    # Plot component waveform
    evokeds = [component[id].average() for id in component.event_id]
    mne.viz.plot_compare_evokeds(evokeds, axes=ax_component, show=False,
                                 truncate_xaxis=False, truncate_yaxis=False, show_sensors=False)
    ax_component.set_title('Waveform')
    # Adjust plots
    try:
        fig.suptitle(f'Subject {epochs.info['temp']}, Component #{component_n}')
    except:
        fig.suptitle(f'Component #{component_n}')
    fig.tight_layout()
    plt.show()
    
def plot_component_topo(A:np.array, info:mne.Info, axes=None):
    '''
    Plot spatial pattern topomap
    
    Parameters
    ----------
    A
        Spatial projection -- np array size of n_channels
    info
        mne.Info instance
    
    '''
    
    component = mne.EvokedArray(np.expand_dims(A,1), info)
    component.plot_topomap(times=0, colorbar=False, cmap='jet', show=False, axes=axes)

class SpatialFilterFC():
    '''
    A class that allows to perform spatial filtration
    with Fisher criterion on the EEG data. Includes
    forward and invers transform, plotting tools.
    
    
    '''
    
    
    def __init__(self, info:mne.Info=None):
        
        self.info = info
    
    def fit(self, epochs:mne.Epochs=None, tmin:float=None, tmax:float=None,
            X:np.ndarray=None, Y:np.ndarray=None, equalize_epoch_counts=False, return_WdA=False):
        
        '''Pipeline to calculate spatial filters
    
        Parameters
        ----------
        * epochs -- mne.Epochs instance with 2 event classes
        * X -- np.ndarray data of class 1
        * Y -- np.ndarray data of class 2
        
        Returns
        -------
        * W - spatial filters
        * d - eigenvalues,
        * A - sparital projectioins
            
        
        '''
        
        if epochs:
            assert len(epochs.event_id.keys()) == 2, 'Must provide epochs with 2 event classes picked'
            self.info = epochs.info.copy()
            
            # Alter epochs
            ep = epochs.copy().crop(tmin, tmax)
            evidn, evidy = ep.event_id.keys()
            ep1, ep2 = ep[evidn], ep[evidy]
            if equalize_epoch_counts:
                mne.epochs.equalize_epoch_counts([ep1, ep2])
            
            # Prepare arrays
            X, Y = ep1.get_data(copy=True), ep2.get_data(copy=True)
            X = np.transpose(X, (1, 2, 0)) # (ch x time x epochs)
            Y = np.transpose(Y, (1, 2, 0)) # (ch x time x epochs)
        
        # Calculater spatial filters
        W, d, A = FCsf(X, Y)
        
        self.W = W
        self.d = d
        self.A = A
        
        if return_WdA:
            return W, d, A
    
    def apply(self, epochs:mne.Epochs=None, pick_component=None):
        '''Apply spatial filter to epochs
    
        Parameters
        ----------
        * epochs: mne.Epochs instance
        * X: np.ndarray
        * pick_component: id of the component to apply to the data
        
        Returns
        -------
        Adjusted epochs
        
        '''
        
        X = epochs.get_data(copy=True)
        X = np.transpose(X,(0,2,1)) # Ch to last dim
        Xf = X @ self.W[pick_component]
        info = mne.create_info(ch_names=['ch'], ch_types=['eeg'], sfreq=1000)
        
        return mne.EpochsArray(np.expand_dims(Xf,1), info,
                               event_id=epochs.event_id, events=epochs.events,
                               tmin=epochs.tmin, baseline=epochs.baseline)
    
    def apply_inverse(self, epochs:mne.Epochs, drop_components_ids):
        '''Perform the inverse transformation for sf
        
        Parameters
        ----------
        * epochs: mne.Epochs instance
        * drop_components: int or list of ints of components to remove from the data
        
        Returns
        -------
        Modified epochs
        
        '''
    
        # Prepare epochs
        X = epochs.get_data(copy=True)
        X = X.transpose((0, 2, 1)) # epochs x times x channels
        # Prepare matrices
        W1 = np.delete(self.W, drop_components_ids, 0)
        A1 = np.delete(self.A, drop_components_ids, 0)
        # Inverse transform
        Xt = (X @ W1.T) @ A1
        Xt = Xt.transpose(0, 2, 1) # epochs x channels x times
        
        return mne.EpochsArray(Xt, epochs.info, epochs.events, event_id=epochs.event_id,
                               tmin=epochs.tmin, baseline=epochs.baseline)
    
    def _plot_topo(self, component_id, axes=None):
        
        '''
        Plot spatial pattern topomap
        
        Parameters
        ----------
        * A: Spatial projection
        * info: mne.Info instance
        
        '''
        
        if not axes: axes = plt.subplot()
        
        component = mne.EvokedArray(np.expand_dims(self.A[component_id],1), self.info)
        component.plot_topomap(times=0, colorbar=False, cmap='jet', show=False, axes=axes)
        axes.set_title('')
    
    def plot_topomap(self, component_ids):
        '''Plot several components'''
        
        n = len(component_ids)
        assert n<=5, 'Can plot only 5 components at a time'
        
        fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
        for i, ax in zip(component_ids, axes.ravel()):
            self._plot_topo(i, ax)
            ax.set_title(f'{i}')
        
        plt.show()
    
    def plot_component_signal(self, epochs:mne.Epochs, component_id:int, axes=None, show=True):
        '''Plot waveform of a given component for a given data'''
        
        # Apply filter
        component = self.apply(epochs, component_id)
        # Calculate evokeds
        evokeds = [component[id].average() for id in component.event_id]
        
        # Plot data
        if not axes: axes = plt.subplot()
        
        mne.viz.plot_compare_evokeds(evokeds, axes=axes, show=False,
                                     truncate_xaxis=False, truncate_yaxis=False, show_sensors=False)
        
        axes.set_title('Waveform')
        
        if show: plt.show()
    
    def plot_component(self, epochs:mne.Epochs, component_id:int):
        '''Plot topomap and waveform of a given component'''
        
        fig, axes = plt.subplots(1,2, width_ratios=(1,2), figsize=(9,3))
        ax_sp, ax_component = axes
        
        self._plot_topo(component_id, ax_sp)
        ax_sp.set_title('Spatial pattern')
        self.plot_component_signal(epochs, component_id, ax_component, show=False)
        
        fig.tight_layout()
        
        plt.show()
        
    def plot_eigenvalues(self):
        '''Plot eigenvalues'''
        
        ax = plt.subplot()
        sns.lineplot(self.d, ax=ax)
        sns.scatterplot(self.d, ax=ax)
        ax.set(xlabel='Component N', ylabel='Eigenvalue')
        
        plt.show()
        