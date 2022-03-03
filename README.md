# py_RMAP
###  RMAP connectivity correlation decoding


This toolbox allows calculation of connectome based correlation analysis.
The correlation map of several connectivity files (structural or functional) in nifti format are estimated with a given effect, this could be symptom improvement [[1]](#1), decoding performance [[2]](#2) and many others.


<img src="./img/ConnectomePrediction.jpg" alt="ConnectomePrediction" width="450"/>

Figure adaped by [[1](#1)]

### Installation
Fist clone the repo:
```
git clone https://github.com/neuromodulation/py_RMAP.git
```

The create a new conda environment:
```
conda env create --file=env.yml
```

then activate and install it:
```
conda activate RMAP
conda develop .
```

### Usage
```
from py_RMAP import RMAP

rmap = RMAP.RMAP()

fingerprint_name, fingerprint_list = RMAP.get_fingerprints_from_path_with_cond(
    path_dir, keep = True, str_to_keep="sub_000",
    connectivity_name_str="_AvgR_Fz.nii"
)

fp_arr = RMAP.convert_to_arr(fingerprint_list)

correlate_performances = np.load(...)

rmap_arr = RMAP.get_RMAP_np(fp_arr, correlate_performances)

RMAP.save_nii(rmap_arr, affine_transform, reshape=True)

```

### References
<a id="1">[1]</a> 
Li and Hollunder (2021). 
A Unified Functional Network Target for Deep Brain Stimulation in Obsessive-Compulsive Disorder.
Biological Psychiatry, 90, 710-713.
https://doi.org/10.1016/j.biopsych.2021.04.006

<a id="2">[2]</a> 
Timon Merk (2021). 
Electrocorticography is superior to subthalamic local field potentials for movement decoding in Parkinson’s disease
bioRxiv
https://doi.org/10.1101/2021.04.24.441207
