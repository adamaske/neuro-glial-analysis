import os
import glob
from wrappers.fnirs import fNIRS
#from wrappers.eeg import EEG
#from wrappers.eeg_fnirs import EEG_fNIRS

# NOTE : This pipeline expects a specific folder structure :
# top-level/
#   sub01/
#       trial01/
#           .snirf
#           .hdf5 
# and so on

def locate_experiment_files(folderpath: str):
    """
    Locates .snirf, .hdf5, and creates EEG, fNIRS and EEG_fNIRS objects.

    Args:
        folderpath (str): The root folder containing subject and trial subfolders.
    """
    
    for subject in os.listdir(folderpath):
        subject_path = os.path.join(folderpath, subject)

        if not os.path.isdir(subject_path):
            continue  # Skip non-directory items

        for trial in os.listdir(subject_path):
            trial_path = os.path.join(subject_path, trial)

            if not os.path.isdir(trial_path):
                continue  # Skip non-directory items

            # Locate .snirf file
            snirf_files = glob.glob(os.path.join(trial_path, "*.snirf"))
            snirf_path = snirf_files[0] if snirf_files else None

            # Locate .hdf5 file
            hdf_files = glob.glob(os.path.join(trial_path, "*.hdf5"))
            hdf_path = hdf_files[0] if hdf_files else None

            if snirf_files:
                if len(snirf_files) > 1:
                    print(f"Several .snirf files found in {trial_path}, which do you want to choose?")
                    for idx, path in enumerate(snirf_files):
                        print(f"{idx + 1} : {path}")
                    while True:
                        try:
                            ans = int(input(f" 1-{len(snirf_files)} : "))
                            if 1 <= ans <= len(snirf_files):
                                snirf_path = snirf_files[ans - 1]
                                break
                            else:
                                print("Invalid choice. Please enter a number within the specified range.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                else:
                    snirf_path = snirf_files[0]

            # Choose .hdf5 file
            hdf_files = glob.glob(os.path.join(trial_path, "*.hdf5"))
            hdf_path = None

            if hdf_files:
                if len(hdf_files) > 1:
                    print(f"Several .hdf5 files found in {trial_path}, which do you want to choose?")
                    for idx, path in enumerate(hdf_files):
                        print(f"{idx + 1} : {path}")
                    while True:
                        try:
                            ans = int(input(f" 1-{len(hdf_files)} : "))
                            if 1 <= ans <= len(hdf_files):
                                hdf_path = hdf_files[ans - 1]
                                break
                            else:
                                print("Invalid choice. Please enter a number within the specified range.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                else:
                    hdf_path = hdf_files[0]
            
            print(f"{trial_path} :")
            print(f"snirf_path : ", snirf_path)
            print(f"hdf_path : ", hdf_path)
            fnirs = fNIRS(snirf_path)
            fnirs.print()
            #eeg = EEG(hdf_path)
            #
            #eeg_fnirs = EEG_fNIRS(eeg, fnirs)
            #print("Constructed EEG : ", eeg.print())
            #print("Constructed fNIRS : ", fnirs.print())
            
    

experiment_folder = "daniel_experiment"        
locate_experiment_files(experiment_folder)
# Locate each trial

# Find both EEG and fNIRS file in trial folder
