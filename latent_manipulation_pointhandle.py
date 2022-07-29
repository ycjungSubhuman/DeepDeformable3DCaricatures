import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import tqdm
import io
import numpy as np
import scipy as sp 
import dataset_caricshop3d as dataset
import training_loop_surface as training_loop
import utils, loss, modules, meta_modules

import cyobj.io

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from surface_net import SurfaceDeformationField
from fitting import Optimizer3D, Editing

#----------------------------------------------------------------------------

def _build_map(PATH):
    """
    Build map: index --> path in the dataset
    """
    result = {}
    with open(PATH, 'r') as fin:
        lines = fin.readlines()
    for line in lines:
        tokens = line.split()
        value = ' '.join(tokens[:-2])
        key = int(tokens[-1]) - 1

        result[key] = value 

    return result
    
#----------------------------------------------------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p = configargparse.ArgumentParser()
p.add_argument('--config', required=True, is_config_file=True, help='Evaluation configuration')
p.add_argument('--dir_caricshop', type=str,default='./3dcaricshop', help='3DCaricShop dataset root')
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
p.add_argument('--start_distance', type=float, default=-0.01, help='Start point for manipulation in latent space. (default: -3.0)')
p.add_argument('--end_distance', type=float, default=0.01, help='End point for manipulation in latent space. (default: 3.0)')
p.add_argument('--steps', type=int, default=11, help='Number of steps for image editing. (default: 10)')
p.add_argument('--strength', type=float, default=0.2, help='point edit strength')

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
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name']+'_2dcari')
utils.cond_mkdir(root_path)

print("Point-based editing...")

length = 50 

ds_test = dataset.CaricShop3DTestLandmark_Attr(meta_params['dir_caricshop'])

for s in np.arange(0.0, meta_params['strength']+0.002, 0.002):
    for i in range(1, length+1):
        optim_e = Editing(
            inds_lmk=torch.LongTensor(ds_test.caricshop.landmarks).cuda(),
            model=model,
            V_ref=torch.from_numpy(ds_test.caricshop.V_ref).float().cuda(),
            F=ds_test.caricshop.F,
            w_z=1e5)

        dir_path = os.path.join(root_path, str(i))
        ckpt_path = os.path.join(dir_path, "checkpoints")
        utils.cond_mkdir(ckpt_path)
        latent = torch.Tensor(sp.io.loadmat(os.path.join(root_path, f'{(i):04d}.mat'))['z']) 

        #Edit0: nothing
        V_edit0 = optim_e.decode(latent.cuda()).detach().cpu().numpy()[0]

        #Edit1: longer nose
        inds = [30]
        lmk_edit = np.array([0,0,s])
        z_edit1 = optim_e.run(inds, lmk_edit, latent, np.eye(3), 1.0, np.zeros(3))
        V_edit1 = optim_e.decode(z_edit1).detach().cpu().numpy()[0]

        #Edit2: longer chin
        inds = [8]
        lmk_edit = np.array([0,-s,0])
        z_edit2 = optim_e.run(inds, lmk_edit, latent, np.eye(3), 1.0, np.zeros(3))
        V_edit2 = optim_e.decode(z_edit2).detach().cpu().numpy()[0]

        #Edit3: wider
        inds = [77, 78]
        lmk_edit = np.zeros((2,3)) 
        lmk_edit[0] += np.array([-s, 0, 0])
        lmk_edit[1] += np.array([s, 0, 0])
        z_edit3 = optim_e.run(inds, lmk_edit, latent, np.eye(3), 1.0, np.zeros(3))
        V_edit3 = optim_e.decode(z_edit3).detach().cpu().numpy()[0]

        #Edit4: bigger ear
        inds = [72, 75]
        lmk_edit = np.zeros((2,3)) 
        lmk_edit[0] += np.array([-s, 0, 0])
        lmk_edit[1] += np.array([s, 0, 0])
        z_edit4 = optim_e.run(inds, lmk_edit, latent, np.eye(3), 1.0, np.zeros(3))
        V_edit4 = optim_e.decode(z_edit4).detach().cpu().numpy()[0]

        V, F, VT, FT, _, _ = cyobj.io.read_obj(os.path.join(root_path, f'{(i):04d}.obj'))
        texture_path = os.path.join(root_path, f'{(i):04d}_tex.png')

        for j in range(0, 5) :
            cyobj.io.write_obj(
                os.path.join(ckpt_path, f'{(i):04d}_edit{j}_{s}.obj'),
                globals()[f'V_edit{j}'].astype(np.double),
                F,
                VT=VT,
                FT=FT,
                path_img=texture_path,
            )
