import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import igl
import yaml
import tqdm
import io
import numpy as np
import dataset_caricshop3d as dataset
import training_loop_surface as training_loop
from scipy.io import savemat
from scipy.spatial.transform import Rotation
import utils, loss, modules, meta_modules
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from cyobj.io import write_obj
import skimage.io as sio

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from surface_net import SurfaceDeformationField
# from calculate_chamfer_distance import compute_recon_error
from fitting import OptimizerBaseline, AlternatingLandmarkOptimizer, bc2vind

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p = configargparse.ArgumentParser()
p.add_argument('--config', required=True, is_config_file=True, help='Evaluation configuration')
p.add_argument('--dir_caricshop', type=str,default='./3dcaricshop', help='3DCaricShop dataset root')
p.add_argument('--dir_caricature_data', type=str,default='./Caricature-Data', help='Alive Caricature test dataset root')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--summary_root', type=str, default='./summaries', help='root for summary')
p.add_argument('--checkpoint_path', type=str, default='', help='checkpoint to use for eval')
p.add_argument('--experiment_name', type=str, default='default',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=256, help='training batch size.')
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--epochs', type=int, default=8000, help='Number of epochs to train for.')

p.add_argument('--epochs_til_checkpoint', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--latent_dim', type=int,default=128, help='latent code dimension.')
p.add_argument('--hidden_num', type=int,default=128, help='hidden layer dimension of deform-net.')
p.add_argument('--num_hidden_layers', type=int,default=3, help='number of hidden layers of deform-net.')
p.add_argument('--hyper_hidden_layers', type=int,default=1, help='number of hidden layers hyper-net.')

# load configs
opt = p.parse_args()
meta_params = vars(opt)

# define DIF-Net
model = SurfaceDeformationField(1268, **meta_params)
model.load_state_dict(torch.load(meta_params['checkpoint_path']))
# The network should be fixed for evaluation.

if hasattr(model, 'hyper_net'):
    for param in model.hyper_net.parameters():
        param.requires_grad = False
model.cuda()

# create save path
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name']+'_2dcari_81')
utils.cond_mkdir(root_path)

ds_test = dataset.Landmarkonly_data(meta_params['dir_caricshop'], meta_params['dir_caricature_data'])

F = ds_test.caricshop.F

def deform_soft(V, F, b, bc, w):
    """
    Deform shape with soft constraints
    
    V       N x dim original shape
    F       F x 3
    b       C position constraint vertex indices 
    bc      C x dim position constraints.
    w       C position constraint weight
    
    Returns)
    Deformed shape N x dim
    """
    C = igl.cotmatrix(V, F)
    data = np.ones(b.shape[0])
    ii = np.arange(b.shape[0])
    jj = b
    I = sp.sparse.coo_matrix((data, (ii, jj)), shape=[b.shape[0], C.shape[1]]).tocsr()
    W = sp.sparse.diags(w)
    
    A = sp.sparse.vstack([C, W@I])
    
    l0 = C @ V
    
    b = np.vstack([l0, W @ bc])
    
    ATA = A.T @ A
    ATb = A.T @ b
    
    x = sp.sparse.linalg.spsolve(ATA, ATb)
    
    return x

for i, (lmk, lmk_flip)in enumerate(ds_test):

    optim = OptimizerBaseline(
        model=model, 
        V_ref=torch.from_numpy(ds_test.caricshop.V_ref).float().cuda(),
        F=ds_test.caricshop.F,
        w_z = 1e7
        )
    alter = AlternatingLandmarkOptimizer(optim, ds_test.caricshop.horlines_81)
    alter.run(lmk_flip)

    pose = (alter.s, alter.R, alter.t)

    V_canon = alter.V.detach().cpu().numpy()[0]
    V_pose = pose[0] * V_canon @ pose[1].T + pose[2]
    error = np.mean(np.linalg.norm(alter.get_landmarks()[:,:2] - lmk, axis=1))
    lmks_bc = alter.lmks_bc
    vinds = ds_test.caricshop.landmarks
    z = alter.z.detach().cpu().numpy()

    lmk_flip_3d = np.zeros([lmk_flip.shape[0], 3])
    lmk_flip_3d[:,0] = lmk_flip[:,0]
    lmk_flip_3d[:,1] = lmk_flip[:,1]
    lmk_flip_3d[:,2] = V_pose[vinds, 2]
    # First stretch the fitting to remove errors
    V_stretch = deform_soft(V_pose, F, vinds, lmk_flip_3d, np.ones(81))

    metadata = {
        'V_canon': V_canon,
        'V_pose': V_pose,
        'error': error,
        'lmks_bc': lmks_bc,
        'z': z,
    }
    
    # Save results
    path_V_pose = os.path.join(root_path, '{:04d}.obj'.format(i+1))
    path_metadata = os.path.join(root_path, '{:04d}.mat'.format(i+1))
    path_vis = os.path.join(root_path, '{:04d}.png'.format(i+1))
    path_image = os.path.join(root_path, '{:04d}_tex.png'.format(i+1))

    igl.write_obj(path_V_pose, V_pose, ds_test.caricshop.F)
    igl.path_metadata = savemat(path_metadata, metadata)

