import h5py


filepath = "data/Subject02/Trial 3/2025-03-27_004.snirf"

hdf = h5py.File(filepath, mode="r+")
print(hdf.keys())

nirs_data = hdf["nirs"]
print(nirs_data.keys())

metadata = nirs_data["metaDataTags"]
print(metadata.keys())

data = nirs_data["data1"]
print(data.keys())