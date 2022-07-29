import os
import igl
import torch
import pickle
import numpy as np
import os.path as osp
from transform import Affine3D, align_3d_to_2d, ralign

from dataset_caricshop3d import CaricShop3D
from sklearn.linear_model import Ridge

def vind2bc(vinds, F):
    lmks = []
    finds = []
    bcs = []
    for ind in vinds:
        target = np.argwhere(F == ind)[0]
        bc = np.array([0.0, 0.0, 0.0])
        
        find = target[0]
        bc[target[1]] = 1.0
        
        lmks.append((find, bc))
    return lmks


def bc2vind(lmks, F):
    vinds = []
    for find, bc in lmks:
        maxind = np.argmax(bc)
        vind = F[find, maxind]
        vinds.append(vind)
    return np.array(vinds, dtype=np.int32)


def get_lmk_pos(lmks_bc, V, F):
    finds = np.array([find for find, _ in lmks_bc], dtype=np.int32)
    bcs = np.stack([bc for _, bc in lmks_bc])
    pos_lmk_Vp = np.einsum('Ni,Nij->Nj', bcs, V[F[finds]])

    return pos_lmk_Vp


class AlternatingLandmarkOptimizer:
    def __init__(self, optim, horlines, iters=4):
        """
        optim_shape: (R, s, t, lmk) --> shape, param
        """
        self.optim = optim
        self.horlines = horlines
        if type(optim.k) == int:
            self.z = torch.zeros([1, self.optim.k]).float().to(self.optim.device)
        else:
            self.z = torch.zeros([1, self.optim.k[0]+self.optim.k[1]]).float().to(self.optim.device)
        self.V = self.optim.decode(self.z)
        self.lmks_bc = [line[0] for line in horlines]
        self.iters = iters

        self.R = np.eye(3)
        self.s = 1.0
        self.t = np.zeros(3)
    
    def _update_landmarks(self):
        V = self.V.detach().cpu().numpy()[0] @ self.R.T
        F = self.optim.F
        lmks_bc_new = []
        axis = np.array([0.0, 0.0, 1.0])
        for line in self.horlines:
            dots = []
            for find, bc in line:
                pos = bc @ V[F[find]]
                dot = (axis @ pos)**2
                dots.append(dot)
            dots = np.array(dots)
            selection = np.argmin(dots)
            lmks_bc_new.append(line[selection])
        self.lmks_bc = lmks_bc_new

    def get_landmarks(self):
        V = self.s * self.V.detach().cpu().numpy()[0] @ self.R.T + self.t
        F = self.optim.F
        result = []
        for find, bc in self.lmks_bc:
            pos = bc @ V[F[find]]
            result.append(pos)
        return np.stack(result)

    def get_landmarks_p(self):
        V = self.V.detach().cpu().numpy()[0] @ self.R.T
        F = self.optim.F
        result = []
        for find, bc in self.lmks_bc:
            pos = bc @ V[F[find]]
            result.append(pos)
        return np.stack(result)

    def _get_pose(self, lmk):
        V = self.V.detach().cpu().numpy()[0]
        pos_lmk = get_lmk_pos(self.lmks_bc, V, self.optim.F)
        self.R, self.s, self.t = align_3d_to_2d(pos_lmk, lmk)

    def run(self, lmk):
        for _ in range(self.iters):
            with torch.no_grad():
                self._get_pose(lmk)
                self._update_landmarks()
            self.z = self.optim.run(lmk, self.lmks_bc, self.z, self.R, self.s, self.t)
            with torch.no_grad():
                self.V = self.optim.decode(self.z)

class OptimizerBaseline:
    def __init__(self, model, 
            V_ref, F, k=128,
            lr=1e-4, criteria=1e-10, device='cuda',
            max_iter=100000, w_z=1e6):
        self.model = model.to(device)
        self.k = k
        self.lr = lr
        self.criteria = criteria
        self.device = device
        self.max_iter = max_iter
        self.w_z = w_z

        self.V_ref = V_ref
        self.F = F

    def decode(self, z):
        result = self.model.inference(self.V_ref, z)
        return result

    def run(self, lmk, lmks_bc, z_init, R, s, t):
        z = z_init.detach().clone().to(self.device).requires_grad_()
        opt = torch.optim.Adam([z], lr=self.lr)
        z.requires_grad_(True)

        finds = torch.LongTensor([find for find, _ in lmks_bc]).to(self.device)
        bcs = torch.stack([torch.from_numpy(bc) for _, bc in lmks_bc]).float().to(self.device)

        R = torch.from_numpy(R).float().to(self.device)
        s = s
        t = torch.from_numpy(t).float().to(self.device)
        F = torch.from_numpy(self.F).long()
        lmk = torch.from_numpy(lmk).float().to(self.device).unsqueeze(0)
        
        error_prev = 1000000000
        for i in range(self.max_iter):
            opt.zero_grad()
            V_new = self.model.inference_back(self.V_ref, z)
            Vp_canon = V_new
            Vp = s * Vp_canon @ R.T + t

            pos_lmk_Vp = torch.einsum('Ni,BNij->BNj', bcs, Vp[:, F[finds]])[:,:,:lmk.shape[2]]
            loss = \
                3e3*torch.nn.functional.mse_loss(pos_lmk_Vp, lmk) / s \
                    + self.w_z*torch.sum(z*z)
            loss.backward()
            opt.step()
            
            if i % 1000 == 0:
                print(loss.item())
                
            error = loss.item()
            delta = error_prev - error
            if delta / error < self.criteria:
                print(loss.item())
                print('Early')
                break
            error_prev = loss.item()
        return z

class SimpleOptimizerManager:
    def __init__(self, optim):
        """
        optim_shape: (R, s, t, lmk) --> shape, param
        """
        self.optim = optim
        self.z = torch.zeros([1, self.optim.k]).float().to(self.optim.device)

        self.V = self.optim.decode(self.z)

    def _get_pose(self, lmk):
        V = self.V.detach().cpu().numpy()[0]
        pos_lmk = self.get_landmarks()
        self.R, self.s, self.t = ralign(pos_lmk, lmk)

    def get_landmarks(self):
        V = self.V.detach().cpu().numpy()[0]
        F = self.optim.F
        return V[self.optim.inds_lmk.detach().cpu().numpy()]

    def run(self, lmk):
        self._get_pose(lmk)
        self.z = self.optim.run(lmk, self.z, self.R, self.s, self.t)
        self.V = self.optim.decode(self.z)

class Optimizer3D:
    def __init__(self, model, 
            V_ref, F, inds_lmk, k=128,
            lr=1e-4, criteria=1e-10, device='cuda',
            max_iter=100000, w_z=1e6):
        self.model = model.to(device)
        self.k = k
        self.lr = lr
        self.criteria = criteria
        self.device = device
        self.max_iter = max_iter
        self.w_z = w_z
        self.inds_lmk = torch.Tensor(inds_lmk).long().to(device)

        self.V_ref = V_ref
        self.F = F

    def decode(self, z):
        result = self.model.inference(self.V_ref, z)
        return result

    def run(self, lmk, z_init, R, s, t):
        z = z_init.detach().clone().to(self.device).requires_grad_()
        opt = torch.optim.Adam([z], lr=self.lr)
        z.requires_grad_(True)

        R = torch.from_numpy(R).float().to(self.device)
        s = s
        t = torch.from_numpy(t).float().to(self.device)

        lmk = torch.from_numpy(lmk).float().to(self.device).unsqueeze(0)
        error_prev = 1000000000
        for i in range(self.max_iter):
            opt.zero_grad()
            V_new = self.model.inference_back(self.V_ref, z)
            Vp_canon = V_new
            Vp = s * Vp_canon @ R.T + t

            pos_lmk_Vp = Vp[:,self.inds_lmk,:]
            loss = \
                3e3*torch.nn.functional.mse_loss(pos_lmk_Vp, lmk) / s \
                    + self.w_z*torch.sum(z*z)
            loss.backward()
            opt.step()
            
            if i % 1000 == 0:
                print(loss.item())
                
            error = loss.item()
            delta = error_prev - error
            if delta / error < self.criteria:
                print(loss.item())
                print('Early')
                break
            error_prev = loss.item()
        return z


class Editing:
    def __init__(self, model, 
            V_ref, F, inds_lmk, k=128,
            lr=1e-4, criteria=1e-10, device='cuda',
            max_iter=100000, w_z=1e6):
        self.model = model.to(device)
        self.k = k
        self.lr = lr
        self.criteria = criteria
        self.device = device
        self.max_iter = max_iter
        self.w_z = w_z
        self.inds_lmk = inds_lmk

        self.V_ref = V_ref
        self.F = F

    def decode(self, z):
        result = self.model.inference(self.V_ref, z)
        return result

    def run(self, inds_vertex, pos_new, z_init, R, s, t):
        z_0 = z_init.clone().to(self.device)
        z = z_init.detach().clone().to(self.device).requires_grad_()
        opt = torch.optim.Adam([z], lr=self.lr)
        z.requires_grad_(True)

        inds_vertex = torch.Tensor(inds_vertex).long().to(self.device)

        R = torch.from_numpy(R).float().to(self.device)
        s = s
        t = torch.from_numpy(t).float().to(self.device)

        pos_new = torch.from_numpy(pos_new).float().to(self.device).unsqueeze(0)
        error_prev = 1000000000

        V_new = self.model.inference(self.V_ref, z)
        V_orig = V_new.clone()[:,self.inds_lmk[inds_vertex],:]

        for i in range(self.max_iter):
            opt.zero_grad()
            V_new = self.model.inference_back(self.V_ref, z)
            Vp_canon = V_new
            #Vp = s * Vp_canon @ R.T + t
            Vp = Vp_canon

            pos_lmk_Vp = Vp[:,self.inds_lmk[inds_vertex],:]
            loss = \
                3e3*torch.nn.functional.mse_loss(pos_lmk_Vp, V_orig + pos_new) / s \
                    + self.w_z*torch.nn.functional.l1_loss(z, z_0)
            loss.backward()
            opt.step()
            
            if i % 1000 == 0:
                print(loss.item())
                
            error = loss.item()
            delta = error_prev - error
            if error == 0 or delta / error < self.criteria:
                print(loss.item())
                print('Early')
                break
            error_prev = loss.item()
        return z
