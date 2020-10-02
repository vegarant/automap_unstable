import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import numpy as np;
from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.automap_tools import read_automap_k_space_mask, compile_network, hand_f, hand_dQ;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, scale_to_01
from PIL import Image
from scipy.io import loadmat

use_gpu = False
compute_node = 3
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


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

im_nbr_gauss = 0;
im_nbr_poisson = 1;


N = 128
size_zoom = 80

new_im1 = mpimg.imread(join(src_data, 'brain1_128_anonymous.png'))
new_im2 = mpimg.imread(join(src_data, 'brain2_128_anonymous.png'))

mri_data = np.zeros([2, N,N], dtype='float32')
mri_data[0, :, :] = new_im1
mri_data[1, :, :] = new_im2

batch_size = mri_data.shape[0]

plot_dest = './plots_random'
data_dest = './data_random'

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest)
if not (os.path.isdir(data_dest)):
    os.mkdir(data_dest)

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)

# Create noise, mri_data.shape = [2,N,N]
noise_gauss = np.float32(np.random.normal(loc=0, scale=1, size=mri_data.shape))
noise_poisson = np.float32(np.random.poisson(lam=2, size=mri_data.shape))

# Scale the noise
norm_mri_data_gauss = l2_norm_of_tensor(mri_data[im_nbr_gauss])
norm_mri_data_poisson = l2_norm_of_tensor(mri_data[im_nbr_poisson])
norm_noise_gauss = l2_norm_of_tensor(noise_gauss[im_nbr_gauss])
norm_noise_poisson = l2_norm_of_tensor(noise_poisson[im_nbr_poisson])

print('mri_data.shape: ', mri_data.shape);

p = 0.02;
noise_gauss *= (p*norm_mri_data_gauss/norm_noise_gauss);
noise_poisson *= (p*norm_mri_data_poisson/norm_noise_poisson);

# Save noise
fname_data = 'noise_%d_automap.mat' % (round(1000*p));

scipy.io.savemat(join(data_dest, fname_data), {'noise_gauss': noise_gauss, 'noise_poisson': noise_poisson});

image_noisy_gauss = mri_data + noise_gauss
image_noisy_poisson = mri_data + noise_poisson

fx_noise_gauss = f(image_noisy_gauss);
fx_noise_poisson = f(image_noisy_poisson);

for i in range(batch_size):
    # Save reconstruction with noise
    image_data_gauss = np.uint8(255*scale_to_01(fx_noise_gauss[i]));
    image_data_poisson = np.uint8(255*scale_to_01(fx_noise_poisson[i]));

    image_rec_gauss = Image.fromarray(image_data_gauss);
    image_rec_poisson = Image.fromarray(image_data_poisson);

    image_rec_gauss.save(join(plot_dest, 'im_gauss_rec_p_%d_nbr_%d.png' % (round(p*1000), i)));
    image_rec_poisson.save(join(plot_dest, 'im_poisson_rec_p_%d_nbr_%d.png' % (round(p*1000), i)));

    # Save original image with noise
    image_orig_gauss = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_gauss[i]))));
    image_orig_poisson = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_poisson[i]))));

    image_orig_gauss.save(join(plot_dest, 'im_gauss_noise_p_%d_nbr_%d.png' % (round(p*1000), i)));
    image_orig_poisson.save(join(plot_dest, 'im_poisson_noise_p_%d_nbr_%d.png' % (round(p*1000), i)));

    # Create zoomed crops, reconstructions 
    image_rec_gauss_zoom = Image.fromarray(image_data_gauss[:size_zoom, -size_zoom:]);
    image_rec_poisson_zoom = Image.fromarray(image_data_poisson[:size_zoom, -size_zoom:]);

    image_rec_gauss_zoom.save(join(plot_dest, 'im_gauss_rec_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));
    image_rec_poisson_zoom.save(join(plot_dest, 'im_poisson_rec_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));

    # Create zoomed crops, images with noise 
    image_orig_gauss_zoom = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_gauss[i, :size_zoom, -size_zoom:]))));
    image_orig_poisson_zoom = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_poisson[i, :size_zoom, -size_zoom:]))));

    image_orig_gauss_zoom.save(join(plot_dest, 'im_gauss_noise_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));
    image_orig_poisson_zoom.save(join(plot_dest, 'im_poisson_noise_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));

sess.close();

