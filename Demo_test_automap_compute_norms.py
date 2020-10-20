"""
This script computes the norms of all quantities required to check the conditions 
of Theorem 1. 
"""

import time

import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path

from PIL import Image


from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, l1_norm_of_tensor, linf_norm_of_tensor, scale_to_01
from adv_tools_PNAS.Runner import Runner;
from adv_tools_PNAS.Automap_Runner import Automap_Runner;
from adv_tools_PNAS.automap_tools import load_runner, read_automap_k_space_mask, compile_network, hand_f, sample_image, adjoint_of_samples, extract_null_space, remove_null_space;
from scipy.io import loadmat
import matplotlib.pyplot as plt

runner_id_automap = 5;

dest_plots = 'plots_automap2';

#if not (os.path.isdir(dest_data)):
#    os.mkdir(dest_data);

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);

N = 128

samp = np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool))
k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

extract_null_space1 = lambda im: extract_null_space(im, k_mask_idx1, k_mask_idx2);
remove_null_space1 = lambda im: remove_null_space(im, k_mask_idx1, k_mask_idx2);


runner = load_runner(runner_id_automap);

im_nbr = 2;
mri_data = runner.x0[0] 
print('mri_data.shape: ', mri_data.shape )
print('samp.shape: ', samp.shape )

batch_size = 1;
mri_data.shape[0];
start = time.time()

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

f = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)
g = lambda x: scale_to_01(f(x))

sample = lambda im: sample_image(im, k_mask_idx1, k_mask_idx2)


def compute_norms_v2_lp(f, image, x_prime, p=2, runner_id=runner_id_automap):
    X_rec_noiseless = scale_to_01(f(image));
    x_prime2 = scale_to_01(f(x_prime));
    print(f"---------------- Runner ID: {runner_id_automap} --------------------")
    #print("runner ID: ", runner_id_automap)
    #print("r_value: ", r_value)
    if p == 2:
        norm_tensor = lambda x: l2_norm_of_tensor(x)
    elif p == 1:
        norm_tensor = lambda x: l1_norm_of_tensor(x)
    elif p.lower() == 'inf':
        norm_tensor = lambda x: linf_norm_of_tensor(x)
    
    print(f"|x|_{p}: ", norm_tensor(image), f', |f(Ax) - x|_{p}: ', norm_tensor(image-X_rec_noiseless))
    print(f"|x-x''|_{p}: ", norm_tensor(image-x_prime2));
    print(f"|x'|_{p}: ", norm_tensor(x_prime), f", |x'' - f(Ax')|_{p}: 0")
    print(f"|f(Ax) - f(Ax')|_{p}: ", norm_tensor(X_rec_noiseless - x_prime2))
    print(f"|A(x-x')|_{p}: ", norm_tensor(sample(image-x_prime)) )
    print(f"|Ax|_{p}:", norm_tensor(sample(image)))
    print('Bound_ratio: ', norm_tensor(X_rec_noiseless-x_prime2)/(norm_tensor(image-x_prime2)- 2*norm_tensor(image-X_rec_noiseless)))


def compute_norms_v2_lp(f, image, x_prime,r_value, p=2, runner_id=runner_id_automap):
    X_rec_noiseless = scale_to_01(f(image));
    x_prime2 = scale_to_01(f(x_prime));
    print(f"---------------- Runner ID: {runner_id_automap} --------------------")
    #print("runner ID: ", runner_id_automap)
    #print("r_value: ", r_value)
    if p == 2:
        norm_tensor = lambda x: l2_norm_of_tensor(x)
    elif p == 1:
        norm_tensor = lambda x: l1_norm_of_tensor(x)
    elif p.lower() == 'inf':
        norm_tensor = lambda x: linf_norm_of_tensor(x)

    n_im = norm_tensor(image);
    n_im_prime = norm_tensor(x_prime);
    n_Aim = norm_tensor(sample(image));
    n_Aim_prime = norm_tensor(sample(x_prime));
    n_diff_im_xpp = norm_tensor(image-x_prime2);
    n_diff_im_rec = norm_tensor(image-X_rec_noiseless);
    n_diff_fim_fxp = norm_tensor(X_rec_noiseless - x_prime2);
    n_diff_Aim_Axp = norm_tensor(sample(image - x_prime))

    print(f"|x|_{p}: ", n_im, f', |f(Ax) - x|_{p}: ', n_diff_im_rec)
    print(f"|x-x''|_{p}: ", n_diff_im_xpp);
    print(f"|x'|_{p}: ", norm_tensor(x_prime), f", |x'' - f(Ax')|_{p}: 0")
    print(f"|f(Ax) - f(Ax')|_{p}: ", n_diff_fim_fxp)
    print(f"|A(x-x')|_{p}: ", n_diff_Aim_Axp )
    print(f"|Ax|_{p}:", n_Aim)
    print('Bound_ratio: ', n_diff_fim_fxp/(n_diff_im_xpp - n_diff_im_rec))

    print("")
    print(" j     |xj'|     |Axj'|   |Ax-Ax_j|     |f(Ax)-f(Axj)|   |x-xj''|   |x-\psi(Ax)|     ratio")
    print(f"${r_value}$ & ${n_im_prime:.1f}$ & {n_Aim_prime:.1f} & ${n_diff_Aim_Axp:.1f}$ & $ {n_diff_fim_fxp:.1f}  & ${n_diff_im_xpp:.1f}$ & ${n_diff_im_rec:.1f}$ & ${n_diff_fim_fxp/(n_diff_im_xpp - n_diff_im_rec):.2f}$ \\\\")




im_nbr = 2
image = mri_data[im_nbr];
image = np.expand_dims(image, 0);

for r_value in range(1,5):
    rr = runner.r[r_value];
    rr = rr[im_nbr, :, :];
    rr = np.expand_dims(rr, 0)
    print('np.amax((image+rr)): ', np.amax(abs(image+rr)));
    compute_norms_v2_lp(f, image, image+rr, r_value, p=2);

