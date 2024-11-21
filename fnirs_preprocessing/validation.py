def validate_snirf(snirf):
    """
    Validate a snirf file. 
    Wrapper for pysnirf2.validateSnirf. 
    If snirf object sent, a temporary 

    Args:
        snirf (mne.raw or str) : Either snirfFilepath to ".snirf" file.
    Returns:
        valid (bool) : True for valid snirf. 
    """
    from mne_nirs.io import write_raw_snirf
    from snirf import validateSnirf
    from os import remove

    if isinstance(snirf, str): #
        result = validateSnirf(snirf)
        result.display()
        return result.is_valid()
    
    temp_filepath = "temp/filepath" 
    write_raw_snirf(snirf, "temporary.snirf")
    print("Validation : Created temporary file : ", temp_filepath)
    
    result = validateSnirf(temp_filepath)
    result.display()

    try:
        remove(temp_filepath)
        print("Validation : Deleted temporary file ", temp_filepath)
    except FileNotFoundError:
        print("Validation : Cannot delete temporary (FileNotFound) : ", temp_filepath)
    except PermissionError:
        print("Validation : Cannot delete temporary (PermissionError) : ", temp_filepath)
    except Exception as e:
        print("Validation : Error occured : ", temp_filepath)

    return result.is_valid()
