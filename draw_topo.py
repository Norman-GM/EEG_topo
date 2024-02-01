import mne
import numpy as np
import matplotlib.pyplot as plt

SEED_CHANNEL_LIST = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'PO10', 'O1', 'Oz', 'O2', 'O9'
]

def draw_topo(weight):
    weight = weight[0]
    delta, theta, alpha, beta, gamma = weight[:62], weight[62:124], weight[124:186], weight[186:248], weight[248:310]
    rol = 1
    col = 5
    fig, axes = plt.subplots(rol, col, figsize=(12, 4))
    for ax, col in zip(axes, ['delta', 'theta', 'alpha', 'beta', 'gamma']):
        ax.set_title(col, {'fontsize': 30}, pad=30)
    ch_types = ['eeg'] * len(SEED_CHANNEL_LIST)
    info = mne.create_info(ch_names=SEED_CHANNEL_LIST, ch_types=ch_types, sfreq=256)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    for i, (label) in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma']):
        norm = plt.Normalize(weight.min(), weight.max())
        mne.viz.plot_topomap(weight[i * 62:(i + 1) * 62], info, axes=axes[i], show=False,cnorm=norm)
    fig.tight_layout()
    return fig
    # plt.show()