# Example Python program that creates a hierarchy of groups
# and datasets in a HDF5 file using h5py

import os
import cv2
import h5py
import glob
import random
import numpy as np
import numpy.random
import natsort


def generate_3_channel_mask(mask_arr):
    num_channels = 3
    m = mask_arr[:, :, 0]
    mask_3c = np.zeros(mask_arr.shape)

    for chnl in range(num_channels):
        blank_mask = np.zeros(m.shape)
        blank_mask[m == chnl] = 1
        mask_3c[:,:, chnl] = blank_mask

    return mask_3c

def main():
    # Init
    eye_side = "R"
    IS_ERROR = False
    dataset_folder = "E:\\Data\\V7 Annotations\\03_08_2021\\emilia-4kplr-0308\\EL_PLR_4K"
    hierarchicalFileName = os.path.join(os.path.dirname(__file__), f"train_{os.path.basename(dataset_folder)}_{eye_side}.hdf5")
    hierarchicalFile = h5py.File(hierarchicalFileName, "w")

    # Create a group under root
    root_grp = hierarchicalFile

    # Get list of all images and mask files
    img_list = sorted(glob.glob(os.path.join(dataset_folder, "images", f"*_{eye_side}.*")), key=os.path.getmtime)
    img_list = natsort.natsorted(img_list, reverse=False)

    mask_list = sorted(glob.glob(os.path.join(dataset_folder, "masks", f"*_{eye_side}.*")), key=os.path.getmtime)
    mask_list = natsort.natsorted(mask_list, reverse=False)

    if (len(img_list) != len(mask_list)) and (os.path.basename(img_list[-1]) != os.path.basename(mask_list[-1])):
        IS_ERROR = True
        print("ERROR: Mismatch in image v/s mask filelist")

    if not IS_ERROR:

        # Create another dataset inside the same group
        (C, D, H, W) = (3, len(img_list), 160, 224)
        datasetShape = (C, D, H, W)
        dataset_label = root_grp.create_dataset("label", datasetShape)
        dataset_raw = root_grp.create_dataset("raw", datasetShape)

        # Print the HDF5 structure
        print(hierarchicalFile["/"]); print(root_grp)
        print(root_grp["label"]); print(root_grp["raw"])

        # Add image data to the 'raw' dataset
        for d, img_path in enumerate(img_list):
            img = np.array(cv2.imread(img_path), dtype=np.int8)
            # temp_img = np.expand_dims(img, axis=2)
            # temp_img = np.einsum('klji->ijkl', temp_img)
            temp_img = np.einsum('jki->ijk', img)
            dataset_raw[:, d, :, :] = temp_img

        mask_list = mask_list[-10:]

        # Add mask data to the 'label' dataset
        for d, mask_path in enumerate(mask_list):
            mask = np.array(cv2.imread(mask_path), dtype=np.int16)
            mask_3c = generate_3_channel_mask(mask)
            temp_mask_3c = np.einsum('jki->ijk', mask_3c)
            dataset_label[:, d, :, :] = temp_mask_3c

# main()

# Code to read a sample.h5 file
import h5py
filename = "E:\\Source\\pytorch-3dunet\\Ovules_Dataset\\Train\\Movie1_t00003_crop_gt.h5"
# filename = "E:\\Source\\pytorch-3dunet\\resources\\sample_ovule.h5"
# filename = "E:\\Source\\pytorch-3dunet\\create_hdf5\\train_EL_PLR_4K_L.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])