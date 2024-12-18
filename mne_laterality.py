import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from mne.io import read_raw_snirf
import mne
import os
import pandas as pd
left_hemisphere_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 34, 38, 39, 40, 41, 42, 43, 44]
right_hemisphere_channels = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 45, 45, 46, 47, 48, 49, 51]
motor_roi = ( #Motor cortex
        ["S1-D1", "S2-D1", "S1-D3", "S2-D3", "S9-D1", "S1-D9", "S10-D2", "S3-D3", "S3-D2" ] ,  #LEFT HEMISPHEre
        ["S5-D5", "S5-D6", "S5-D7", "S5-D9", "S9-D5", "S6-D5", "S6-D7", "S6-D7", "S10-D6" ] #RIGHT HEMISPHERES
)

#Load GLM results
omid_path = "data/glm_results/omid"
omid_filepaths = []
runs = []
run_names = []
for entry in os.listdir(omid_path):
        file_path = os.path.join(omid_path, entry)
        runs.append(pd.read_excel(file_path))
        run_names.append("Run " + str(len(runs)))
        omid_filepaths.append(file_path)

#Identify channel indices
channels = {}
for channel in left_hemisphere_channels + right_hemisphere_channels:
        ch_name = str(runs[0].iloc[channel, 1]) 
        channels[ch_name] = channel
#load snirfs
snirf_paths = [
       #"data/balance-22-11/2024-11-22_006/2024-11-22_006.snirf",
        "data/OMID-13-12-024/2024-12-13_001/2024-12-13_001.snirf", 
        "data/OMID-13-12-024/2024-12-13_002/2024-12-13_002.snirf",
        "data/OMID-13-12-024/2024-12-13_003/2024-12-13_003.snirf",
        "data/OMID-13-12-024/2024-12-13_004/2024-12-13_004.snirf",
        ]

snirfs = []
for path in snirf_paths:
        snirf = read_raw_snirf(path).load_data()
        snirfs.append(snirf)

raw_intensity = read_raw_snirf(snirf_paths[2]).load_data()
raw_intensity.annotations.rename({"0":"REST", "1":"RIGHT", "2":"LEFT", "3":"END"})
raw_intensity.annotations.set_durations({"REST":20, "RIGHT":10, "LEFT":10, "END":1})

#REMOVE REST AND END
unwanted = np.nonzero(raw_intensity.annotations.description == "REST")
raw_intensity.annotations.delete(unwanted)
unwanted = np.nonzero(raw_intensity.annotations.description == "END")
raw_intensity.annotations.delete(unwanted)


# SKIP THIS : we're using all channels atm
#picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
#dists = mne.preprocessing.nirs.source_detector_distances(
#    raw_intensity.info, picks=picks
#)
#raw_intensity.pick(picks[dists > 0.01])
#raw_intensity.plot(
#    n_channels=len(raw_intensity.ch_names), duration=500, show_scrollbars=False
#)

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity) # to od
#raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)

#CHECKS SCALP COUPLING INDEX -> OMID DATA IS EXPENTIONALLY CLEAN
sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))

#DISPLAY SCALP COUØING
#fig, ax = plt.subplots(layout="constrained")
#ax.hist(sci)
#ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])

raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

#FILTERING
#raw_haemo_unfiltered = raw_haemo.copy()
raw_haemo.filter(0.02, 0.07, h_trans_bandwidth=0.05, l_trans_bandwidth=0.01)
#for when, _raw in dict(Before=raw_haemo_unfiltered, After=raw_haemo).items():
#    fig = _raw.compute_psd().plot(
#        average=True, amplitude=False, picks="data", exclude="bads"
#    )
#    fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
#raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=500, show_scrollbars=False)

events, event_dict = mne.events_from_annotations(raw_haemo)
#fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_haemo.info["sfreq"])

tmin, tmax = -2, 15
epochs = mne.Epochs(
    raw_haemo,
    events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    reject=None,
    reject_by_annotation=True,
    proj=True,
    baseline=(None, 0),
    preload=True,
    detrend=None,
    verbose=True,
)
#epochs.plot_drop_log()

#epochs["RIGHT"].plot_image(
#    combine="mean",
#    vmin=-50,
#    vmax=50,
#    ts_args=dict(ylim=dict(hbo=[-30, 30], hbr=[-30, 30])),
#)
#epochs["LEFT"].plot_image(
#    combine="mean",
#    vmin=-50,
#    vmax=50,
#    ts_args=dict(ylim=dict(hbo=[-30, 30], hbr=[-30, 30])),
#)
evoked_dict = {
    "LEFT/HbO": epochs["LEFT"].average(picks="hbo"),
    "LEFT/HbR": epochs["LEFT"].average(picks="hbr"),
    "RIGHT/HbO": epochs["RIGHT"].average(picks="hbo"),
    "RIGHT/HbR": epochs["RIGHT"].average(picks="hbr"),
}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO="#AA3377", HbR="b")
styles_dict = dict(RIGHT=dict(linestyle="dashed"))

mne.viz.plot_compare_evokeds(
    evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict
)

#times = np.arange(-3.5, 13.2, 3.0)
topomap_args = dict(extrapolate="local")
#epochs["LEFT"].average(picks="hbr").plot_joint(
#    times=times, topomap_args=topomap_args
#)
#epochs["RIGHT"].average(picks="hbr").plot_joint(
#    times=times, topomap_args=topomap_args
#)

times = np.arange(4.0, 11.0, 1.0)
#epochs["LEFT"].average(picks="hbo").plot_topomap(times=times, **topomap_args)
#epochs["RIGHT"].average(picks="hbo").plot_topomap(times=times, **topomap_args)
#epochs["LEFT"].average(picks="hbr").plot_topomap(times=times, **topomap_args)
#epochs["RIGHT"].average(picks="hbr").plot_topomap(times=times, **topomap_args)

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(9, 5),
    gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]),
    layout="constrained",
)
vlim = (-8, 8)
ts = 9.0

evoked_left = epochs["LEFT"].average()
evoked_right = epochs["RIGHT"].average()

evoked_left.plot_topomap(
    ch_type="hbo", times=ts, axes=axes[0, 0], vlim=vlim, colorbar=False, **topomap_args
)
evoked_left.plot_topomap(
    ch_type="hbr", times=ts, axes=axes[1, 0], vlim=vlim, colorbar=False, **topomap_args
)
evoked_right.plot_topomap(
    ch_type="hbo", times=ts, axes=axes[0, 1], vlim=vlim, colorbar=False, **topomap_args
)
evoked_right.plot_topomap(
    ch_type="hbr", times=ts, axes=axes[1, 1], vlim=vlim, colorbar=False, **topomap_args
)

evoked_diff = mne.combine_evoked([evoked_left, evoked_right], weights=[1, -1])

evoked_diff.plot_topomap(
    ch_type="hbo", times=ts, axes=axes[0, 2:], vlim=vlim, colorbar=True, **topomap_args
)
evoked_diff.plot_topomap(
    ch_type="hbr", times=ts, axes=axes[1, 2:], vlim=vlim, colorbar=True, **topomap_args
)

for column, condition in enumerate(["RIGHT", "LEFT", "LEFT-RIGHT"]):
    for row, chroma in enumerate(["HbO", "HbR"]):
        axes[row, column].set_title(f"{chroma}: {condition}")

plt.show()

exit()

raw_intensity.annotations.set_durations(5)
raw_intensity.annotations.rename(
    {"1.0": "Control", "2.0": "Tapping/Left", "3.0": "Tapping/Right"}
)
unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
raw_intensity.annotations.delete(unwanted)

picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks
)
raw_intensity.pick(picks[dists > 0.01])
raw_intensity.plot(
    n_channels=len(raw_intensity.ch_names), duration=500, show_scrollbars=False
)

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)

sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
fig, ax = plt.subplots(layout="constrained")
ax.hist(sci)
ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])

raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))

raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=500, show_scrollbars=False)
