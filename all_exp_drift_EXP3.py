import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

import functions_titanicks_DRIFT as fn
import functions_fitting_DRIFT as fit


# Initialise arrays
delta1_a_999, delta1_b_999, delta1_c_999 = [], [], []
delta1_a_987, delta1_b_987, delta1_c_987 = [], [], []

all_mse = []
# Find directories for all experiments
exp_folder = Path('drift_exp_v2')
experiments_pos = [x for x in exp_folder.iterdir() if x.is_dir()]

for exp_path in tqdm(experiments_pos, desc='Drift Exp'):
    print(exp_path)
    exp_path_str = str(exp_path)
    exp_type = exp_path_str[exp_path_str.rfind('_U_')+3:exp_path_str.rfind('_U_')+8]

    if exp_type == 'ph8_8': print(F'EXP {exp_path_str}: ph8-8')
    elif exp_type == 'ph8_9': print(F'EXP {exp_path_str}: ph8-9')
    else: print(F'EXP {exp_path_str}: ERROR. COULD NOT MATCH EXPERIMENT TYPE')

    # LOAD
    time, chem_mean, chem_2d_active = fn.load_and_preprocessing_DRIFT(exp_path, input_type='bin', version='v4', plt_summary=False, save_summary=False, print_status=False)

    # SPLIT SEGMENTS
    t1, x1, t2, x2, dist1_idx = fit.split_2_segments(chem_2d_active, time, show_plt=False)

    # INITIALISE ARRAYS FOR TMP PARAMETERS rows are xmin-xmax-xo'-t0-a-b
    param_seg1 = np.zeros((6, chem_2d_active.shape[1]))
    param_seg2 = np.zeros((6, chem_2d_active.shape[1]))
    param_seg1[0, :] = np.min(x1, axis=0)
    param_seg1[1, :] = np.max(x1, axis=0)
    param_seg1[2, :] = np.mean(x1[:3, :], axis=0)
    param_seg1[3, :] = t1[0]

    param_seg2[0, :] = np.min(x2, axis=0)
    param_seg2[1, :] = np.max(x2, axis=0)
    param_seg2[2, :] = np.mean(x2[:3, :], axis=0)
    param_seg2[3, :] = t2[0]

    # FIT ALL SEGMENTS
    # t is always from 0; x is always bs
    param_seg1[4:6, :] = fit.fit_pixels_interpolate_EXP2(t1 - t1[0], x1 - np.mean(x1[:3, :], axis=0))
    param_seg2[4:6, :] = fit.fit_pixels_interpolate_EXP2(t2 - t2[0], x2 - np.mean(x2[:3, :], axis=0))

    for i_pixel in range(param_seg1.shape[1]):
        mse = np.mean((fit.decaying_exp(t1 - t1[0], param_seg1[4, i_pixel], param_seg1[5, i_pixel]) - (x1[:,i_pixel] - np.mean(x1[:3, i_pixel], axis=0)))**2)
        all_mse.append(mse)

    a1 = param_seg1[4, :]+param_seg1[2, :]
    b1 = param_seg1[4, :]*np.exp(param_seg1[3, :]/param_seg1[5, :])
    c1 = param_seg1[5, :]

    a2 = param_seg2[4, :]+param_seg2[2, :]
    b2 = param_seg2[4, :]*np.exp(param_seg2[3, :]/param_seg2[5, :])
    c2 = param_seg2[5, :]

    # FIND DELTAS
    if exp_type == 'ph8_8':
        delta1_a_999.append((a2-a1))
        delta1_b_999.append((b2-b1))
        delta1_c_999.append((c2-c1))
    elif exp_type == 'ph8_9':
        delta1_a_987.append((a2-a1))
        delta1_b_987.append((b2-b1))
        delta1_c_987.append((c2-c1))
    else:
        print('experiment type is not defined. results not saved. ')

all_mse_mean = np.mean(mse)
print(f"AMM MSE {all_mse_mean}")

# PLOT DELTA COEFFICIENTS
fig, ax = plt.subplots(3, 1, figsize=(5.5, 5.5), dpi=200)
fig.suptitle('Cumulative exponential fit results')
color88, color89 = [], []
for i in range(len(delta1_a_999)):
    color88.append('g')
    color89.append('r')
ax[0].hist(delta1_a_999,range=(-5,5), bins=50, density=True, histtype='bar', stacked=True, color=color88, alpha=0.5, label='pH8-8')
ax[0].hist(delta1_a_987,range=(-5,5), bins=50, density=True, histtype='bar', stacked=True, color=color89, alpha=0.5, label='pH8-9')
ax[0].set(title=f'a', xlabel=r'$|a_2|-|a_1|$', ylabel='count')
ax[0].legend()
ax[0].grid(linestyle='--', color=(230/255, 230/255, 230/255))

ax[1].hist(delta1_b_999,range=(-5,5), bins=50, density=True, histtype='bar', stacked=True, color=color88, alpha=0.5, label='8-8')
ax[1].hist(delta1_b_987,range=(-5,5), bins=50, density=True, histtype='bar', stacked=True, color=color89, alpha=0.5, label='8-9')
ax[1].set(title=f'b', xlabel=r'$|b_2|-|b_1|$', ylabel='count')
ax[1].legend()
ax[1].grid(linestyle='--', color=(230/255, 230/255, 230/255))

ax[2].hist(delta1_c_999,range=(-5, 5), bins=50, density=True, histtype='bar', stacked=True, color=color88, alpha=0.5, label='8-8')
ax[2].hist(delta1_c_987,range=(-5, 5), bins=50, density=True, histtype='bar', stacked=True, color=color89, alpha=0.5, label='8-9')
ax[2].set(title=f'c', xlabel=r'$|c_2|-|c_1|$', ylabel='count')
ax[2].legend()
ax[2].grid(linestyle='--', color=(230/255, 230/255, 230/255))

plt.tight_layout()
# plt.savefig('drift_pics\exponential_allexp.png')
plt.show()
