import pathlib
import os 
import snirf

dir_path = os.path.dirname(os.path.realpath(__file__)) #path of this file

def data_folder():
    return os.path.join(os.getcwd(), "data")

def is_snirf_file(file_path): #Check for snirf file
    if pathlib.Path(file_path).suffix != '.snirf':
        return False

    result = snirf.validateSnirf(file_path)
    return result.is_valid()

def find_snirf_in_folder(folder_path):
    """
    Locate all .snirf files in directory and subdirectories. 
    Args:
        folder_path (str) : Path to directory.
    Returns:
        snirf_paths (list) : List containg filepaths to all located .snrif files.
    """
    snirf_paths = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)

        if os.path.isdir(entry_path):# is this a folder?
            snirfs = find_snirf_in_folder(entry_path)
            for file in snirfs:
                snirf_paths.append(file)
            continue

        if is_snirf_file(entry_path): #is this a snirf file?
            snirf_paths.append(entry_path)

    return snirf_paths
