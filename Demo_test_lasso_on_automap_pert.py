"""
This test the LASSO algorithm on the adversarial perturbation computed for the 
AUTOMAP network.
"""

import time

import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path

from optimization.gpu.operators import MRIOperator
from optimization.gpu.proximal import WeightedL1Prox, SQLassoProx2
from optimization.gpu.algorithms import SquareRootLASSO
from optimization.utils import estimate_sparsity, generate_weight_matrix
from tfwavelets.dwtcoeffs import get_wavelet
from tfwavelets.nodes import idwt2d
from PIL import Image

from adv_tools_PNAS.automap_config import src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor

from adv_tools_PNAS.Runner import Runner;
from adv_tools_PNAS.Automap_Runner import Automap_Runner;
from adv_tools_PNAS.automap_tools import load_runner;


runner_id_automap = 5;

N = 128
wavname = 'db2'
levels = 3
use_gpu = True
compute_node = 2
dtype = tf.float64;
sdtype = 'float64';
scdtype = 'complex128';
cdtype = tf.complex128
wav = get_wavelet(wavname, dtype=dtype);

if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    print('Compute node: {}'.format(compute_node))
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

# Turn on soft memory allocation
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
sess = tf.compat.v1.Session(config=tf_config)


dest_data = 'data_lasso';
dest_plots = 'plots_lasso';

if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);



# Parameters for CS algorithm
pl_sigma = tf.compat.v1.placeholder(dtype, shape=(), name='sigma')
pl_tau   = tf.compat.v1.placeholder(dtype, shape=(), name='tau')
pl_lam   = tf.compat.v1.placeholder(dtype, shape=(), name='lambda')

# Build Primal-dual graph
tf_im = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='image')
tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

# For the weighted l^1-norm
pl_weights = tf.compat.v1.placeholder(dtype, shape=[N,N,1], name='weights')

tf_input = tf_im

op = MRIOperator(tf_samp_patt, wav, levels, dtype=dtype)
measurements = op.sample(tf_input)

tf_adjoint_coeffs = op(measurements, adjoint=True)
adj_real_idwt = idwt2d(tf.math.real(tf_adjoint_coeffs), wav, levels)
adj_imag_idwt = idwt2d(tf.math.imag(tf_adjoint_coeffs), wav, levels)
tf_adjoint = tf.complex(adj_real_idwt, adj_imag_idwt)

prox1 = WeightedL1Prox(pl_weights, pl_lam*pl_tau, dtype=dtype)
prox2 = SQLassoProx2(dtype=dtype)

alg = SquareRootLASSO(op, prox1, prox2, measurements, sigma=pl_sigma, tau=pl_tau, lam=pl_lam, dtype=dtype)

initial_x = op(measurements, adjoint=True)

result_coeffs = alg.run(initial_x)

real_idwt = idwt2d(tf.math.real(result_coeffs), wav, levels)
imag_idwt = idwt2d(tf.math.imag(result_coeffs), wav, levels)
tf_recovery = tf.complex(real_idwt, imag_idwt)




# Parameters for the CS-algorithm
n_iter = 1000
tau = 0.6
sigma = 0.6
lam = 0.0001


samp = np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool))

runner_automap = load_runner(runner_id_automap);

im_nbr = 2;
mri_data = runner_automap.x0[0] 
#print('mri_data.shape: ', mri_data.shape )
#mri_data = np.expand_dims(mri_data, -1)
##mri = np.expand_dims(mri, 0)
#print('mri.shape: ', mri.shape )
samp = np.expand_dims(samp, -1)

batch_size = mri_data.shape[0];
start = time.time()
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    weights = np.ones([128,128,1], dtype=sdtype);

    bd = 5;
    N = 128;
    for idx in range(len(runner_automap.r)):
        rr = runner_automap.r[idx];
        if idx == 0:
            rr = np.zeros(rr.shape, dtype=rr.dtype);

        noisy_image = mri_data + rr;
        noisy_image = noisy_image[im_nbr];
        noisy_image = np.expand_dims(noisy_image, -1)
        noisy_image = noisy_image.astype(scdtype);

        np_im_rec = sess.run(tf_recovery, feed_dict={'tau:0': tau,
                                                     'lambda:0': lam,
                                                     'sigma:0': sigma,
                                                     'weights:0': weights,
                                                     'n_iter:0': n_iter,
                                                     'image:0': noisy_image,
                                                     'sampling_pattern:0': samp})

        fname_out_data = f'im_rID_{runner_id_automap}_im_nbr_{im_nbr}_pert_nbr_{idx}_rec'
        np.save(join(dest_data, fname_out_data), np_im_rec);

        fname_out_rec   =  f'im_auto_pert_rID_{runner_id_automap}_im_nbr_{im_nbr}_pert_nbr_{idx}_rec.png'
        fname_out_noisy = f'im_auto_pert_rID_{runner_id_automap}_im_nbr_{im_nbr}_pert_nbr_{idx}_orig_p_noise.png'

        Image_im_rec = Image.fromarray(np.uint8(255*np.abs(np.squeeze(np_im_rec))));
        Image_im_orig = Image.fromarray(np.uint8(255*np.abs(np.squeeze(noisy_image))));

        Image_im_rec.save(join(dest_plots, fname_out_rec));
        Image_im_orig.save(join(dest_plots, fname_out_noisy));


