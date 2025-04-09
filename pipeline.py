from neuropipeline.fnirs import fNIRS
from neuropipeline.eeg import EEG


hdf5_supination = ["data/Subject01/Trial 1 - Supination/heel2025.03.24_14.27.28.hdf5",
                   "data/Subject01/Trial 3 - Supination/heel2025.03.24_14.36.01.hdf5",
                   "data/Subject01/Trial 5 - Supination/heel2025.03.24_14.45.30.hdf5",
                   "data/Subject02/Trial 5/HeelSubject22025.03.27_11.29.06.hdf5",
                   "data/Subject02/Trial 3/HeelSubject22025.03.27_11.21.29.hdf5",
                   "data/Subject02/Trial 1/HeelSubject22025.03.27_11.14.27.hdf5",
]

hdf5_pronation = ["data/Subject01/Trial 6 - Pronation/heel2025.03.24_14.50.12.hdf5",
                  "data/Subject01/Trial 4 - Pronation/heel2025.03.24_14.40.18.hdf5",
                  "data/Subject01/Trial 2 - Pronation/heel2025.03.24_14.31.33.hdf5",
                  "data/Subject02/Trial 2/HeelSubject22025.03.27_11.17.47.hdf5",
                  "data/Subject02/Trial 4/HeelSubject22025.03.27_11.25.31.hdf5",
                  "data/Subject02/Trial 6/HeelSubject22025.03.27_11.32.54.hdf5",
]

snirf_supination = [
"data/Subject01/Trial 1 - Supination/2025-03-24_001.snirf",
"data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf",
"data/Subject01/Trial 5 - Supination/2025-03-24_005.snirf",
"data/Subject02/Trial 1/2025-03-27_002.snirf",
"data/Subject02/Trial 3/2025-03-27_004.snirf",
"data/Subject02/Trial 5/2025-03-27_006.snirf",
]

snirf_pronation = ["data/Subject01/Trial 2 - Pronation/2025-03-24_002.snirf",
                   "data/Subject01/Trial 4 - Pronation/2025-03-24_004.snirf",
                   "data/Subject01/Trial 6 - Pronation/2025-03-24_006.snirf",
                   "data/Subject02/Trial 2/2025-03-27_003.snirf",
                   "data/Subject02/Trial 4/2025-03-27_005.snirf",
                   "data/Subject02/Trial 6/2025-03-27_007.snirf",
]

fnirs_supination = [fNIRS() for path in snirf_supination]
[f.read_snirf(path) for path, f in zip(snirf_supination, fnirs_supination)]
#fnirs_pronation = [fNIRS(path) for path in snirf_pronation]
#hdf5_supination = [EEG(path) for path in hdf5_supination]
#hdf5_pronation = [EEG(path) for path in hdf5_pronation]

[f.print() for f in fnirs_supination]