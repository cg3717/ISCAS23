# In this file:
# all_exp_drift_LINv2 is validated by using different ph value
# shifting from ph7 to ph8 (at 10 min, and comparing with a control of constant ph8
# in this case, ph7 is replaced by the ph8 solution in the well at 10 min


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

import functions_titanicks_DRIFT as fn
import functions_fitting_DRIFT as fit

# Initialise arrays
experiments_summary = []  # list of dictionaries summarising experiment info

# Find directories for all experiments
exp_folder = Path('drift_exp_v2_validation_ph')
experiments_pos = [x for x in exp_folder.iterdir() if x.is_dir()]

for exp_path in tqdm(experiments_pos, desc='Drift Exp'):
    print(f'Now processing experiment {exp_path}.')
    exp_path_str = str(exp_path)
    exp_type = exp_path_str[exp_path_str.rfind('_U_')+3:exp_path_str.rfind('_U_')+8]
    chip_id = exp_path_str[exp_path_str.rfind('_C') + 2:exp_path_str.rfind('_C') + 4]

    if exp_type == 'ph8_8':
        print(F'EXP {exp_path_str}: ph8-8')
    elif exp_type == 'ph7_8':
        print(F'EXP {exp_path_str}: ph7-8')
    else:
        print(F'EXP {exp_path_str}: ERROR. COULD NOT MATCH EXPERIMENT TYPE {exp_type}')

    # LOAD
    time, chem_mean, chem_2d_active = fn.load_and_preprocessing_DRIFT(exp_path, input_type='bin', version='v4', plt_summary=False, save_summary=False, print_status=False)

    # SPLIT SEGMENTS
    t1, x1, t2, x2, dist1_idx = fit.split_2_segments(chem_2d_active, time, show_plt=False)

    # Calculate mean drift rate of each segment
    mean_drift_rate_1 = np.mean(np.abs(chem_2d_active[dist1_idx - 20, :] - chem_2d_active[50, :])) / (
                time[dist1_idx - 20] - time[50])
    mean_drift_rate_2 = np.mean(np.abs(chem_2d_active[-1, :] - chem_2d_active[dist1_idx + 50, :])) / (
                time[-1] - time[dist1_idx + 50])
    print(f'Mean drift rate in segment 1: {mean_drift_rate_1}; In segment 2: {mean_drift_rate_2}')

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
    param_seg1[4:6, :] = fit.fit_pixels_interpolate_LIN(t1 - t1[0], x1 - np.mean(x1[:3, :], axis=0))
    param_seg2[4:6, :] = fit.fit_pixels_interpolate_LIN(t2 - t2[0], x2 - np.mean(x2[:3, :], axis=0))

    # ADAPT COEFFICIENTS
    a1 = param_seg1[4, :]
    b1 = -param_seg1[4, :]*param_seg1[3, :]+param_seg1[5, :]+param_seg1[2, :]
    a2 = param_seg2[4, :]
    b2 = -param_seg2[4, :]*param_seg2[3, :]+param_seg2[5, :]+param_seg2[2, :]

    # FIND DELTAS
    delta_a = np.abs(a2) - np.abs(a1)
    delta_b = np.abs(b2) - np.abs(b1)
    perc_a = (np.abs(a2) - np.abs(a1)) / np.abs(a1)
    perc_b = (np.abs(b2) - np.abs(b1)) / np.abs(b1)

    # add data to existing dictionary if there is already data for that chip
    exp_summary_stored = False
    for existing_dict in experiments_summary:
        if existing_dict["chip_id"] == chip_id:
            if exp_type == "ph8_8":
                existing_dict["data_88"] = chem_2d_active
                existing_dict["delta_a_88"] = delta_a
                existing_dict["delta_b_88"] = delta_b
                existing_dict["perc_a_88"] = perc_a
                existing_dict["perc_b_88"] = perc_b
                existing_dict["dist_idx_88"] = dist1_idx
                exp_summary_stored = True
            elif exp_type == "ph7_8":
                existing_dict["data_78"] = chem_2d_active
                existing_dict["delta_a_78"] = delta_a
                existing_dict["delta_b_78"] = delta_b
                existing_dict["perc_a_78"] = perc_a
                existing_dict["perc_b_78"] = perc_b
                existing_dict["dist_idx_78"] = dist1_idx
                exp_summary_stored = True
            else:
                print(f'ERROR. COULD NOT MATCH EXPERIMENT TYPE {exp_type}')

    # create new dictionary if data for the chip is not already stored
    if exp_summary_stored == False:
        if exp_type == "ph8_8":
            new_dict = {
                "chip_id": chip_id,
                "data_88": chem_2d_active,
                "delta_a_88": delta_a,
                "delta_b_88": delta_b,
                "perc_a_88": perc_a,
                "perc_b_88": perc_b,
                "dist_idx_88": dist1_idx
            }
            experiments_summary.append(new_dict)
        elif exp_type == "ph7_8":
            new_dict = {
                "chip_id": chip_id,
                "data_78": chem_2d_active,
                "delta_a_78": delta_a,
                "delta_b_78": delta_b,
                "perc_a_78": perc_a,
                "perc_b_78": perc_b,
                "dist_idx_78": dist1_idx
            }
            experiments_summary.append(new_dict)
        else:
            print(f'ERROR. COULD NOT MATCH EXPERIMENT TYPE {exp_type}')

# PLOT EXPERIMENTS SUMMARY
fig, ax = plt.subplots(len(experiments_summary), 2, figsize=(7, 2*len(experiments_summary)), dpi=300)
fig.suptitle('Validation dataset: pH')

plot_id = ['a', 'b', 'c', 'd']
for i, exp_dict in enumerate(experiments_summary):
    ax[i, 0].plot(np.mean(exp_dict["data_88"] - exp_dict["data_88"][0, :], axis=1), c='g', linewidth=2, label='pH8-8')
    ax[i, 0].plot(np.mean(exp_dict["data_78"] - exp_dict["data_78"][0, :], axis=1), c='r', linewidth=2, label='pH7-8')
    ax[i, 0].set(title=f'({plot_id[2*i]}) Mean active pixel data', xlabel='time (s)', ylabel='Vout (mV)')
    ax[i, 0].legend()
    ax[i, 0].grid(linestyle='--', color=(230/255, 230/255, 230/255))

    ax[i, 1].hist(exp_dict["delta_a_88"], range=(-1, 1), bins=100, density=True, label='pH8-8', color='g')
    ax[i, 1].hist(exp_dict["delta_a_78"], range=(-1, 1), bins=100, density=True, label='pH7-8', color='r')
    ax[i, 1].set(title=f'({plot_id[2*i+1]}) CHIP{exp_dict["chip_id"]} - a', xlabel=r'$|a_2|-|a_1|$', ylabel='count')
    ax[i, 1].legend()
    ax[i, 1].grid(linestyle='--', color=(230/255, 230/255, 230/255))

plt.tight_layout()
# plt.savefig('pics_paper/validation_ph_result.png')
plt.show()

# PLOT METRICS FOR COEFFICIENT VARIATION
delta_means_a_88, delta_means_a_78, = [], []
delta_means_b_88, delta_means_b_78, = [], []
perc_means_a_88, perc_means_a_78, = [], []
perc_means_b_88, perc_means_b_78, = [], []
for exp_dict in experiments_summary:
    delta_means_a_88.append(np.mean(exp_dict["delta_a_88"]))
    delta_means_a_78.append(np.mean(exp_dict["delta_a_78"]))

    delta_means_b_88.append(np.mean(exp_dict["delta_b_88"]))
    delta_means_b_78.append(np.mean(exp_dict["delta_b_78"]))

    perc_means_a_88.append(np.mean(exp_dict["perc_a_88"]))
    perc_means_a_78.append(np.mean(exp_dict["perc_a_78"]))

    perc_means_b_88.append(np.mean(exp_dict["perc_b_88"]))
    perc_means_b_78.append(np.mean(exp_dict["perc_b_78"]))

fig, ax = plt.subplots(2,2, figsize=(7,4), dpi=300)
fig.suptitle("Fitted parameter variation across segments for each experiment")
ax[0, 0].vlines(delta_means_a_88, 0, 1, color="g", label="pH8-8")
ax[0, 0].vlines(delta_means_a_78, 0, 1, color="r", label="pH7-8")
ax[0, 0].legend(loc="upper left")
ax[0, 0].set(title=r"(a) $\Delta a$", xlabel=r"$|a_2|-|a_1|$")

ax[0, 1].vlines(delta_means_b_88, 0, 1, color="g", label="pH8-8")
ax[0, 1].vlines(delta_means_b_78, 0, 1, color="r", label="pH7-8")
ax[0, 1].legend(loc="upper left")
ax[0, 1].set(title=r"(b) $\Delta b$", xlabel=r"$|a_2|-|a_1|$")

ax[1, 0].vlines(perc_means_a_88, 0, 1, color="g", label="pH8-8")
ax[1, 0].vlines(perc_means_a_78, 0, 1, color="r", label="pH7-8")
ax[1, 0].legend()
ax[1, 0].set(title=r"(c) %a", xlabel=r"$\frac{|a_2|-|a_1|}{|a_2|}$")

ax[1, 1].vlines(perc_means_b_88, 0, 1, color="g", label="pH8-8")
ax[1, 1].vlines(perc_means_b_78, 0, 1, color="r", label="pH7-8")
ax[1, 1].legend(loc="upper left")
ax[1, 1].set(title=r"(d) %b", xlabel=r"$\frac{|b_2|-|b_1|}{|b_2|}$")

plt.setp(ax, yticks=[0, 1], yticklabels=[])
plt.tight_layout()
plt.show()

print('END.')