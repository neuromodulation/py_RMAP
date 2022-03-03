import os
from typing import Tuple

import numpy as np
import nibabel as nib
from numba import jit


class RMAP:
    def __init__(self) -> None:
        pass

    def load_fingerprint(self, path_nii: str, return_affine: bool = False) -> np.array:
        """Load Nifti data

        Parameters
        ----------
        path_nii : str

        Returns
        -------
        np.array
           Nifi data
        np.array
            Affine transform array
        """

        epi_img = nib.load(path_nii)
        if return_affine is True:
            return epi_img.get_fdata(), epi_img.affine

        return epi_img.get_fdata()

    def get_fingerprints_from_path_with_cond(
        self,
        path_dir: str,
        str_to_omit: str = None,
        str_to_keep: str = None,
        keep: bool = True,
        connectivity_name_str: str = "_AvgR_Fz.nii",
    ) -> Tuple:
        """Read from a directory all Nii Files that contain string connectivity_name_str.

        Additionall when parameter keep is True, str_to_keep files will only be read.
        If keep is False str_to_omit is used to omit files that have str_to_omit in their filename.

        Parameters
        ----------
        path_dir : str
        str_to_omit : str, optional
            by default None
        str_to_keep : str, optional
            by default None
        keep : bool, optional
            by default True
        connectivity_name_str : str, optional
            by default "_AvgR_Fz.nii"

        Returns
        -------
        Tuple
            list with pathnames for each nifti file, list of nifti arrays
        """

        if keep:
            l_fps = list(
                filter(
                    lambda k: connectivity_name_str in k and str_to_keep in k,
                    os.listdir(path_dir),
                )
            )
        else:
            l_fps = list(
                filter(
                    lambda k: connectivity_name_str in k and str_to_omit not in k,
                    os.listdir(path_dir),
                )
            )
        return l_fps, [self.load_fingerprint(os.path.join(path_dir, f)) for f in l_fps]

    def convert_to_arr(self, fingerprint_list: list) -> np.array:
        return np.array([f.flatten() for f in fingerprint_list])

    def save_Nii(
        self,
        fp: np.array,
        affine: np.array = None,
        name_save: str = "img.nii",
        reshape: bool = True,
    ):
        """Save the nifti data fp with affine transform to name_save

        Parameters
        ----------
        fp : np.array
        affine : np.array, optional
            by default None
        name_save : str, optional
            by default "img.nii"
        reshape : bool, optional
            by default True
        """
        if reshape:
            fp = np.reshape(fp, (91, 109, 91), order="F")

        img = nib.nifti1.Nifti1Image(fp, affine=affine)

        nib.save(img, name_save)

    def get_RMAP_np(self, X: np.array, y: np.array) -> np.array:
        """Calculate RMap of array of connectivity files / fingerprints and corresponding correlates y
        Correlation: Pearson's correlation coefficient

        Parameters
        ----------
        X : np.array
            shape (dimension of voxels, number of fingerprints)
        y : np.array
            shape (dimension of voxels)

        Returns
        -------
        np.array

        """
        rmap_calc = (
            len(y) * np.sum(X * y[None, :], axis=-1) - (np.sum(X, axis=-1) * np.sum(y))
        ) / (
            np.sqrt(
                (len(y) * np.sum(X**2, axis=-1) - np.sum(X, axis=-1) ** 2)
                * (len(y) * np.sum(y**2) - np.sum(y) ** 2)
            )
        )
        return rmap_calc

    @staticmethod
    @jit(nopython=True)
    def calculate_RMAP_numba(fp_arr, var_correlate):
        """calculate the RMAP using the numpy corrcoeff function using numba.
        For every voxel, a correlation across all fingerprints and var_correlate is estimated.

        Parameters
        ----------
        fp_arr : np.array
            shape (voxels, number of fingerprints)
        var_correlate : np.array
            shape(number of fingerprints)

        Returns
        -------
        np. array
        """

        NUM_VOXELS, LEN_FPS = fp_arr.shape[0], fp_arr.shape[1]
        fp_arr = np.empty((NUM_VOXELS, LEN_FPS))

        rmap_calc = np.zeros(NUM_VOXELS)
        for voxel in range(NUM_VOXELS):
            corr_val = np.corrcoef(fp_arr[voxel, :], var_correlate)[0][1]
            rmap_calc[voxel] = corr_val

        return rmap_calc

    @staticmethod
    @jit(nopython=True)
    def get_corr_numba(fp, fp_test):
        val = np.corrcoef(fp_test, fp)[0][1]
        return val
