import gc
import os
import random
from typing import Optional, List
from urllib.request import urlretrieve

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic


def load_data(mode: str,
              test_share: float,
              val_share: float,
              seed: int = 42,
              drop_gammas=False,
              binary_classes=False,
              apply_cuts=False,
              digitize=False) -> (np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray,
                                  Optional[List[float]]):
    files_to_download = [f'{mode}_matrices.npz',
                         f'{mode}_features.npz',
                         f'{mode}_true_features.npz']
    for filename in files_to_download:
        os.makedirs('data', exist_ok=True)
        if not os.path.exists(f'data/{filename}'):
            print(f'Downloading {filename}... ', end='')
            urlretrieve(
                f'https://kascade-sim-data.s3.eu-central-1.amazonaws.com/{filename}',
                f'data/{filename}')
            print('Done!')

    matrices = np.load(f'data/{files_to_download[0]}')['matrices']
    features = np.load(f'data/{files_to_download[1]}')['features']
    true_features = np.load(f'data/{files_to_download[2]}')['true_features']

    matrices = matrices[..., 1:]

    if binary_classes:
        # prepare particle id for binary classification
        true_features[:, 1][true_features[:, 1] == 1] = 0
        true_features[:, 1][true_features[:, 1] != 0] = 1
    else:
        true_features[:, 1][true_features[:, 1] == 1] = 0
        true_features[:, 1][true_features[:, 1] == 14] = 1
        true_features[:, 1][true_features[:, 1] == 402] = 2
        true_features[:, 1][true_features[:, 1] == 1206] = 3
        true_features[:, 1][true_features[:, 1] == 2814] = 4
        true_features[:, 1][true_features[:, 1] == 5626] = 5

    # right now we're gonna detect only particle type
    part_class = true_features[:, 1]

    if drop_gammas:
        drop_mask = part_class != 0
        matrices = matrices[drop_mask]
        features = features[drop_mask]
        true_features = true_features[drop_mask]
        part_class = part_class[drop_mask]
        part_class -= 1
    '''
    features: ['part_type', 'E', 'Xc', 'Yc', 'core_dist', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
    true_features: ['E', 'part_type', 'Xc', 'Yc', 'Ze', 'Az', 'Ne', 'Np', 'Nmu', 'Nh']
    '''
    if apply_cuts:
        drop_mask = ((features[:, 5] < 18)
                     * (features[:, 7] > 4.8)
                     * (features[:, 8] > 3.6)
                     * (features[:, 9] < 1.48)
                     * (features[:, 9] > 0.2))
        matrices = matrices[drop_mask]
        features = features[drop_mask]
        true_features = true_features[drop_mask]
        part_class = part_class[drop_mask]

    matrices = matrices.reshape(matrices.shape[0], -1)
    vals = np.unique(true_features[:, [0]], axis=0)
    random_mask = np.random.default_rng(seed).random(vals.shape)
    is_train = np.in1d(true_features[:, [0]],
                       vals[random_mask > (test_share + val_share)])
    is_test = np.in1d(true_features[:, [0]],
                      vals[random_mask < test_share])
    is_val = np.invert(is_train + is_test)
    del random_mask, vals
    matrices_train = matrices[is_train]
    matrices_test = matrices[is_test]
    matrices_val = matrices[is_val]
    del matrices
    if digitize:
        digitize_depth = 1000000
        arr = np.array(random.choices(matrices_train, k=100000))
        splits = np.array_split(np.sort(arr.ravel()), digitize_depth * 2)
        cutoffs = [x[-1] for x in splits][:-1]
        discrete = np.digitize(matrices_train, cutoffs, right=True)
        matrices_train = discrete / digitize_depth - 1
        discrete = np.digitize(matrices_test, cutoffs, right=True)
        matrices_test = discrete / digitize_depth - 1
        discrete = np.digitize(matrices_val, cutoffs, right=True)
        matrices_val = discrete / digitize_depth - 1
        del arr, splits, discrete, digitize_depth
    else:
        cutoffs = None
    class_train = part_class[is_train]
    class_test = part_class[is_test]
    class_val = part_class[is_val]
    features_train = features[is_train]
    features_test = features[is_test]
    features_val = features[is_val]
    true_features_train = true_features[is_train]
    true_features_test = true_features[is_test]
    true_features_val = true_features[is_val]
    del true_features, part_class
    gc.collect()
    return (matrices_train, matrices_test, matrices_val,
            class_train, class_test, class_val,
            features_train, features_test, features_val, cutoffs,
            true_features_train, true_features_test, true_features_val)


def evalute_predictions(test_preds, true_features, title=''):
    test_true = true_features[:, 1]
    test_energy = true_features[:, 0]
    test_theta = true_features[:, 4]

    # Plot figures
    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=(21, 12))
    axs = axs.flatten()
    nbins = 5
    bins_range = (13, 18)
    E = test_energy
    N, energy_bins = np.histogram(E, bins=nbins, range=bins_range)
    energy_bins_centers = energy_bins[1:] - 0.5 * (energy_bins[1] - energy_bins[0])
    energy_bins_half_width = 0.5 * (energy_bins[1] - energy_bins[0])
    # energy_bins_half_width = [0.25] * 2 + [0.5] * (len(nbins) - 3)

    for i, (angle_min, angle_max) in zip(range(3), ((0, 20), (20, 40), (40, 60))):
        cond = np.where(
            (test_theta >= angle_min)
            & (test_theta < angle_max)
            & (test_true == 0)
            & (test_preds == 0))
        correctly_predicted_gamma = binned_statistic(
            E[cond],
            test_preds[cond],
            statistic='count',
            bins=nbins,
            range=bins_range)[0]

        cond = np.where(
            (test_theta >= angle_min)
            & (test_theta < angle_max)
            & (test_preds == 0))
        all_predicted_gamma = binned_statistic(
            E[cond],
            test_preds[cond],
            statistic='count',
            bins=nbins,
            range=bins_range)[0]

        cond = np.where(
            (test_theta >= angle_min) & (test_theta < angle_max) & (test_true == 0))
        all_true_gamma = \
            binned_statistic(E[cond], test_preds[cond], statistic='count',
                             bins=nbins,
                             range=bins_range)[0]

        survival_fraction_gamma = correctly_predicted_gamma / (
                all_true_gamma + 0.001)
        survival_fraction_gamma_error = np.sqrt(correctly_predicted_gamma) / (
                all_true_gamma + 0.001)
        axs[i].errorbar(energy_bins_centers,
                        survival_fraction_gamma,
                        xerr=energy_bins_half_width,
                        yerr=survival_fraction_gamma_error,
                        fmt='o', capsize=3, capthick=3, ms=10, label='$ \\gamma $')

        protons_predicted_as_gamma = all_predicted_gamma - correctly_predicted_gamma
        cond = np.where(
            (test_theta >= angle_min) & (test_theta < angle_max) & (test_true == 1))
        all_true_protons = \
            binned_statistic(E[cond], test_preds[cond], statistic='count',
                             bins=nbins,
                             range=bins_range)[0]

        survival_fraction_protons = protons_predicted_as_gamma / (
                all_true_protons + 0.001)
        survival_fraction_protons_error = np.sqrt(protons_predicted_as_gamma) / (
                all_true_protons + 0.001)
        axs[i].errorbar(energy_bins_centers,
                        survival_fraction_protons,
                        xerr=energy_bins_half_width,
                        yerr=survival_fraction_protons_error,
                        fmt='o', capsize=3, capthick=3, ms=10, label='$p$')

        axs[i].semilogy()
        axs[i].legend(fontsize=15)
        axs[i].xaxis.set_tick_params(labelsize=13)
        axs[i].yaxis.set_tick_params(labelsize=13)
        axs[i].set_xlabel('', fontsize=20)
        axs[i].set_ylabel('survival fraction', fontsize=20)
        axs[i].set_title(f'$ \\theta $ range: {angle_min} - {angle_max} deg',
                         fontsize=15)
    fig.suptitle(title +
                 '\nupper: survival fraction of $ \\gamma $ and p vs energy'
                 '\nlower: energy spectra of the events',
                 fontsize=25)

    for i, (angle_min, angle_max) in zip(range(3, 6), ((0, 20), (20, 40), (40, 60))):
        cond = np.where(
            (test_theta >= angle_min) & (test_theta < angle_max) & (test_true == 0))
        sns.histplot(x=E[cond],
                     bins=nbins,
                     binrange=bins_range,
                     log_scale=(False, False),
                     element='step',
                     fill=False,
                     lw=3,
                     ax=axs[i],
                     label='$ \\gamma $')

        cond = np.where(
            (test_theta >= angle_min) & (test_theta < angle_max) & (test_true == 1))
        sns.histplot(x=E[cond],
                     bins=nbins,
                     binrange=bins_range,
                     log_scale=(False, False),
                     element='step',
                     fill=False,
                     lw=3,
                     ax=axs[i],
                     label='$ p $')

        axs[i].semilogy()
        axs[i].legend(fontsize=15)
        axs[i].xaxis.set_tick_params(labelsize=13)
        axs[i].yaxis.set_tick_params(labelsize=13)
        axs[i].set_xlabel('lg($E_0$/eV)', fontsize=20)
        axs[i].set_ylabel('events', fontsize=20)
    plt.show()
