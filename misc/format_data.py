# Utils to unzip dataset to HPC local storage (/data1)
import tarfile
import os
import shutil

def unzip_data(local_path, dataset_name):
    """    
    Unzip data to local HPC storage location
    """

    # concat paths for specific dataset and create directory
    # data_path = os.path.join(local_path, dataset_name)
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # open file
    file = tarfile.open(f"{os.path.join('../../dataset', dataset_name)}.tar.gz")
    
    # extracting file
    print(f"Extracting dataset to '{local_path}'")
    file.extractall(local_path)
    
    file.close()

    print(f"Successfully extracted dataset")    


def cleanup(local_path):
    """
    Clean local HPC files once finished.
    """
    print(f"Cleaning '{local_path}'")
    shutil.rmtree(local_path)
    print("Successfully cleaned. Exiting")

    




