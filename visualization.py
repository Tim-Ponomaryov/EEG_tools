'''
Tools for EEG visualization
Author: Timofei Ponomarev
ORCID: 0009-0005-0308-8006

'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import mne

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def save_figure(fig, fname, fpath='./results'):
    '''A small wrapper around plt.savefig function'''
    
    fig.savefig(os.path.join(fpath, fname), dpi=300, bbox_inches='tight')

def plot_evokeds_channels(evokeds:dict, title=None, palette='tab10', ax=None, show=True):
    '''Plot ERP with all channels displayed as a separate line'''
    
    palette = sns.color_palette(palette)
    
    evids = list(evokeds.keys())
    times = evokeds[evids[0]].times
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
    
    for i, (evid, evoked) in enumerate(evokeds.items()):
        _ = ax.plot(evoked.get_data().T, c=palette[i] ,lw=0.5, label=evid)
    
    handles, labels = ax.get_legend_handles_labels()
    labdict = dict(zip(labels, handles))
    ax.legend(labdict.values(), labdict.keys())
    sns.despine()
    ax.axhline(0, c='k', lw=0.5)
    ax.axvline(np.where(times==0), c='k', ls='--', lw=0.5)
    ax.set_xlim(0, 1400)
    ax.set_xticks(np.arange(0, 1501,200))
    ax.set_xticklabels(times[np.arange(0, 1501,200)])
    ax.set_title(title)
    
    if show:
        plt.show()

def custom_joint(evoked:mne.Evoked, ax, times, topo_size=1, topo_height=1.5, titles=None):
    '''Joint plot with more flexible configuration
    
    Allows to put joint plot from mne to your custom figure 
    '''
    
    ntimes = len(times)
    devider = make_axes_locatable(ax)
    axD = devider.append_axes('top', 0.5, sharex=None)
    axD.set_visible(False)
    ax_topo = [inset_axes(ax, topo_size, topo_size, bbox_transform=ax.transAxes, bbox_to_anchor=(0.1+i, topo_height)) for i in np.arange(0.5/ntimes, 0.9, 0.9/ntimes)]
    ax_cmap = inset_axes(ax, 0.1, 0.8, bbox_transform=ax.transAxes, bbox_to_anchor=(1, 1.4))
    evoked.plot_joint(times=times, ts_args={'axes':ax}, topomap_args={'axes': ax_topo+[ax_cmap]}, show=False)
    for i, ax_t in enumerate(ax_topo):
        if titles:
            ax_t.set_title(titles[i], pad=0.01)
        else:
            ax_t.set_title('')
    ax_cmap.set_visible(False)
    
def get_ys(amp, pad, n, coef=0.2):
    '''Get y coordinate to plot a significance bar'''
    
    if n==0:
        return amp + coef*pad
    else:
        return get_ys(amp + coef*pad, pad, n-1, coef) 
    
def add_significance(ax, df, sig_levels, positions, metric='amp'):
    '''Add significance bars to and axes'''
    
    n = len(sig_levels)
    pad = ax.get_ylim()[1]-ax.get_ylim()[0]
    _, upper_ylim = ax.get_ylim()
    mx = df[metric].max()
    ax.set_ylim((None, upper_ylim+pad*0.2*n))
    
    pad = ax.get_ylim()[1] - mx
    sig_ys = [get_ys(mx, pad, n, 0.3) for n in range(n)]
    for sig_level, sig_y, pos in zip(sig_levels, sig_ys, positions):
        plot_significance(ax, pos[0], pos[1], sig_y, sig_level=sig_level)

def plot_significance(ax, x1, x2, y, h=None, sig_level=1):
    '''Plot a significance bar on a given axes'''
    
    h = h if h else (ax.get_ylim()[1]-ax.get_ylim()[0])/40
    ax.plot([x1,x1,x2,x2], [y, y+h, y+h,y], lw=1, c='k')
    ax.text((x1+x2)*.5, y+h, '*'*sig_level, ha='center', va='bottom', color='k', fontsize=15)