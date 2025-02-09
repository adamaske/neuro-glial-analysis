
# Load all files
from datasets.fnirs import read_snirf, find_snirf_in_folder
from preprocess import preprocess_snirf
from visualization.fnirs import plot_snirf, plot_psd_snirf

filepaths = find_snirf_in_folder("data/OMID-13-12-024")
files = [read_snirf(f) for f in filepaths]

preprocessed = [preprocess_snirf(f) for f in files]

[plot_snirf(f) for f in preprocessed]
[plot_psd_snirf(f) for f in preprocessed]