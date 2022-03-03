import os
import numpy as np

from py_RMAP import RMAP


def main():

    PATH_FINGERPRINTS = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Connectomics\DecodingToolbox_BerlinPittsburgh_Beijing\functional_connectivity"

    rmap = RMAP.RMAP()

    fingerprint_names, fingerprint_list = rmap.get_fingerprints_from_path_with_cond(
        PATH_FINGERPRINTS,
        keep=True,
        str_to_keep="Berlin",
        connectivity_name_str="_AvgR_Fz.nii",
    )

    # read affine transform from single fingerprint
    _, affine_transform = rmap.load_fingerprint(
        os.path.join(PATH_FINGERPRINTS, fingerprint_names[0]),
        return_affine=True,
    )

    fp_arr = rmap.convert_to_arr(fingerprint_list)

    correlate_performances = np.random.random(fp_arr.shape[0])

    rmap_arr = np.nan_to_num(rmap.get_RMAP_np(fp_arr.T, correlate_performances))

    rmap.save_Nii(rmap_arr, reshape=True, affine=affine_transform)


if __name__ == "__main__":
    main()
