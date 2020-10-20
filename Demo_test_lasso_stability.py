"""
This script run the stablity test on the LASSO algorithm
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
from adv_tools_PNAS.adversarial_tools import scale_to_01, l2_norm_of_tensor
from adv_tools_PNAS.automap_tools import load_runner

dest_data = 'data_lasso';
dest_plots = 'plots_lasso';
count = 4;
dest_data_full = join(dest_data, f'c{count:03}');
dest_plots_full = join(dest_plots, f'c{count:03}');

if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);
if not os.path.isdir(dest_plots_full):
    os.mkdir(dest_plots_full);

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);
if not os.path.isdir(dest_data_full):
    os.mkdir(dest_data_full);

N = 128
wavname = 'db2'
levels = 3
runner_id = 5
use_gpu = True
compute_node = 3
dtype = tf.float64;
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

max_num_noise_iter = 300

# Parameters for Nesterov algorithm
stab_eta = 0.01
stab_gamma = 0.9

# Parameters for the adv. noise objective function
initial_noise_scaling = 1e-3
stab_lambda = 0.01

# Parameters for the CS-algorithm
n_iter = 1000
tau = 0.6
sigma = 0.6
lam = 0.0001


samp = np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool))
samp = np.expand_dims(samp, -1)

im_nbr = 2;

# Compute norms from a spesific runner object and im_nbr
runner = load_runner(runner_id);

rr = runner.r; # List of perturbations
mri_data = runner.x0[0];
mri = mri_data[im_nbr];
mri = np.expand_dims(mri, -1)

norms = [0] # list of norms

for i in range(1,len(rr)):
    pert_all = rr[i]
    pert = pert_all[im_nbr];
    pert_norm = l2_norm_of_tensor(pert);
    norms.append(pert_norm)

print('norms: ', norms)

# The lambda in the objective function for generating adv. noise
pl_noise_penalty = tf.compat.v1.placeholder(dtype, shape=(), name='noise_penalty')
#pl_noise_scaling = tf.compat.v1.placeholder(dtype, shape=(), name='noise_scaling')

# Parameters for CS algorithm
pl_sigma = tf.compat.v1.placeholder(dtype, shape=(), name='sigma')
pl_tau   = tf.compat.v1.placeholder(dtype, shape=(), name='tau')
pl_lam   = tf.compat.v1.placeholder(dtype, shape=(), name='lambda')

# Build Primal-dual graph
tf_im = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='image')
tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')
pl_weights = tf.compat.v1.placeholder(dtype, shape=[N,N,1], name='weights')

# perturbation
tf_rr_real = tf.Variable(tf.constant(initial_noise_scaling, dtype=dtype)*tf.random.uniform(tf_im.shape, dtype=dtype), name='rr_real', trainable=True)
tf_rr_imag = tf.Variable(tf.constant(initial_noise_scaling, dtype=dtype)*tf.random.uniform(tf_im.shape, dtype=dtype), name='rr_imag', trainable=True)

tf_rr = tf.complex(tf_rr_real, tf_rr_imag, name='rr')
tf_input = tf_im + tf_rr

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

tf_solution = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='actual')

tf_obj = tf.nn.l2_loss(tf.abs(tf_recovery - tf_solution)) - pl_noise_penalty * tf.nn.l2_loss(tf.abs(tf_rr))
# End building objective function for adv noise



opt = tf.compat.v1.train.MomentumOptimizer(stab_eta, stab_gamma, use_nesterov=True).minimize(
        -tf_obj, var_list=[tf_rr_real, tf_rr_imag])

fileID = open(join(dest_data, 'stability_test_info.txt'), 'a+');

start = time.time()
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    #weights = generate_weight_matrix(mri.shape[0], estimate_sparsity(mri, wavname, levels), np.float32)
    #weights = np.expand_dims(weights, -1)
    weights = np.ones([128,128,1], dtype='float32');
    print('weights.shape: ', weights.shape);
    noiseless = sess.run(tf_recovery, feed_dict={ 'tau:0': tau,
                                                  'lambda:0': lam,
                                                  'sigma:0': sigma,
                                                  'weights:0': weights,
                                                  'n_iter:0': n_iter,
                                                  'image:0': mri,
                                                  'sampling_pattern:0': samp,
        })


    scipy.io.savemat(join(dest_data_full, f'im_rID_{runner_id}_im_nbr_{im_nbr}_experi_{count:03}_noiseless.mat'), 
            {'image': mri, 'image_rec': noiseless})

    fname_out_rec  = join(dest_plots_full, f'im_rID_{runner_id}_im_nbr_{im_nbr}_rec_noiseless.png'); 
    fname_out_orig = join(dest_plots_full, f'im_rID_{runner_id}_im_nbr_{im_nbr}_orig_noiseless.png'); 

    Image_im_rec_noiseless = Image.fromarray(np.uint8(255*scale_to_01(np.abs(np.squeeze(noiseless)))));
    Image_im_orig = Image.fromarray(np.uint8(255*scale_to_01(np.abs(np.squeeze(mri)))));

    Image_im_rec_noiseless.save(fname_out_rec);
    Image_im_orig.save(fname_out_orig);

    i = 1;
    length = -1;
    pert_nbr = 0;
    max_norm = norms[pert_nbr]
    while(i < (max_num_noise_iter+1) and length < max_norm):

        sess.run(opt, feed_dict={'image:0': mri,
                                 'sampling_pattern:0': samp,
                                 'sigma:0': sigma,
                                 'tau:0': tau,
                                 'lambda:0': lam,
                                 'n_iter:0': n_iter,
                                 'noise_penalty:0': stab_lambda,
                                 'weights:0': weights,
                                 'actual:0': noiseless})

        rr = sess.run(tf.complex(tf_rr_real, tf_rr_imag))

        im_adjoint = sess.run(tf_adjoint, feed_dict={'image:0': mri,
                                                         'sampling_pattern:0': samp,
                                                         'sigma:0': sigma,
                                                         'lambda:0': lam,
                                                         'tau:0': tau,
                                                         'n_iter:0': n_iter,
                                                         'noise_penalty:0': stab_lambda,
                                                         'weights:0': weights,
                                                         'actual:0': noiseless})

        im_rec = sess.run(tf_recovery, feed_dict={'image:0': mri,
                                                     'sampling_pattern:0': samp,
                                                     'sigma:0': sigma,
                                                     'tau:0': tau,
                                                     'n_iter:0': n_iter,
                                                     'weights:0': weights,
                                                     'noise_penalty:0': stab_lambda,
                                                     'lambda:0': lam,
                                                     'actual:0': noiseless})

        rr_save = np.squeeze(rr);
        im_adjoint = np.squeeze(im_adjoint);
        im_rec = np.squeeze(im_rec);
        im_orig_p_noise = np.squeeze(mri+rr);

        length = l2_norm_of_tensor(rr)
        t =  (time.time()-start)/60
        print(f'{i:3}/{max_num_noise_iter}, pert_nbr: {pert_nbr}, Norm: {length:10g} / {max_norm:10g}, time (min): {t:.2f}')

        fileID.write('itr: %3d, Image_nbr: %2d, norm: %g, target norm: %g \n' % (i, pert_nbr, length, max_norm));

        while( length > max_norm and pert_nbr <= len(norms)-1 ): 
            # np.save(join(dest_data, f'im_rID_{runner_id}_im_nbr_{im_nbr}_pert_nbr_{pert_nbr}_adjoint'), im_adjoint)
            # np.save(join(dest_data, f'rr_im_rID_{runner_id}_im_nbr_{im_nbr}_pert_nbr_{pert_nbr}'), rr_save)
            # np.save(join(dest_data, f'im_rID_{runner_id}_im_nbr_{im_nbr}_pert_nbr_{pert_nbr}_rec'), im_rec)
            # np.save(join(dest_data, f'im_rID_{runner_id}_im_nbr_{im_nbr}_pert_nbr_{pert_nbr}_orig_p_noise'), im_orig_p_noise)

            scipy.io.savemat(join(dest_data_full, f'im_rID_{runner_id}_im_nbr_{im_nbr}_experi_{count:03}_pert_nbr_{pert_nbr}.mat'), 
                    {'image': mri, 'image_rec': im_rec, 'rr': rr_save, 'im_adjoint': im_adjoint})

            if pert_nbr == len(norms) - 1:
                i = max_num_noise_iter+1;
            else:
                pert_nbr += 1
            max_norm = norms[pert_nbr];

            fname_out_rec = join(dest_plots_full, f'im_rID_{runner_id}_im_nbr_{im_nbr}_pert_nbr_{pert_nbr}_rec.png')
            fname_out_adjoint = join(dest_plots_full, f'im_rID_{runner_id}_im_nbr_{im_nbr}_pert_nbr_{pert_nbr}_adjoint.png');
            fname_out_orig_p_noise = join(dest_plots_full, f'im_rID_{runner_id}_im_nbr_{im_nbr}_pert_nbr_{pert_nbr}_orig_p_noise.png');

            im_rec = scale_to_01(np.abs(im_rec));
            im_adjoint = scale_to_01(np.abs(im_adjoint));
            im_orig_p_noise = scale_to_01(np.abs(im_orig_p_noise));

            Image_im_rec = Image.fromarray(np.uint8(255*im_rec));
            Image_im_adjoint = Image.fromarray(np.uint8(255*im_adjoint));
            Image_im_orig_p_noise = Image.fromarray(np.uint8(255*im_orig_p_noise));

            Image_im_rec.save(fname_out_rec);
            Image_im_adjoint.save(fname_out_adjoint);
            Image_im_orig_p_noise.save(fname_out_orig_p_noise);

        i += 1;

fileID.close();
