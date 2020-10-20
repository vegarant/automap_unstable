# Deep learning through domain-transform manifold learning for image reconstruction (AUTOMAP) is unstable

Code related to the paper *"Deep learning through domain-transform manifold learning for image reconstruction (AUTOMAP) is unstable"*.

## Setup
The data related to the paper can be downloaded from [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/storage_matters_arising_final.zip). After downloading the data, modify the paths in the file `adv_tools_PNAS/automap_config.py` so that all relevant paths points to the data. To run the stability test for the LASSO experiment, add the [UiO-CS/optimization](https://github.com/UiO-CS/optimization) and [UiO-CS](https://github.com/UiO-CS/tf-wavelets) packages to your Python path. 

## Overview of the different files

----------------------------

* Figure 1: Demo_test_automap_stability.py
* Figure 2: Demo_test_automap_random_noise.py and Demo_test_lasso_random_noise.py
* Figure 3: Demo_test_lasso_stability.py and Demo_test_lasso_on_automap_pert.py
* Figure 4: Demo_test_automap_random_noise.py and Demo_test_lasso_random_noise.py
* SI Table 2: Demo_test_automap_compute_norms.py

---------------------------

All scripts have been exectured with Tensorflow version 1.14.0.


