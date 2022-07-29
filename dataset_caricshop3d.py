import os
import random
import pickle

import torch
import trimesh
import igl

import numpy as np
from glob import glob
from cyobj.io import read_obj
from torch.utils.data import Dataset
from scipy.io import loadmat
import skimage.io as sio
import skimage.filters as sfi

from staticdata import rawlandmarks

class Vertices(Dataset):
    def __init__(self, path_obj, instance_idx, V_ref, F, num_samples=11581, landmarks=None):
        super().__init__()

        self.instance_idx = instance_idx
        self.V_ref = V_ref

        # surface points
        V, F_orig, _, _, _, _ = read_obj(path_obj)
        V = igl.remove_unreferenced(V, F_orig)[0]

        mesh = trimesh.Trimesh(V, F, process=False)
        mesh_ref = trimesh.Trimesh(V_ref, F, process=False)

        self.coords = V_ref
        self.positions = V

        if num_samples > 0:
            V_sample, fids  = trimesh.sample.sample_surface_even(mesh, num_samples)
            bc = trimesh.triangles.points_to_barycentric(mesh.triangles[fids], V_sample)
            V_ref_sample = trimesh.triangles.barycentric_to_points(mesh_ref.triangles[fids], bc)

            self.coords = np.concatenate([V_ref, V_ref_sample], axis=0)
            self.positions = np.concatenate([V, V_sample], axis=0)

        if landmarks is not None:
            print('YESYESYES')
            self.weights = np.ones(self.coords.shape[0])
            #self.weights[landmarks] *= 50
        else:
            self.weights = np.ones(self.coords.shape[0])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'coords': torch.from_numpy(self.coords).float(),
            'positions': torch.from_numpy(self.positions).float(),
            'weight': torch.from_numpy(self.weights).float(),
            'instance_idx':torch.Tensor([self.instance_idx]).squeeze().long()}

class VerticesMulti(Dataset):
    def __init__(self, paths_obj, V_ref, F, num_samples=11581, landmarks=None):
        #This class adapted from SIREN https://vsitzmann.github.io/siren/
        super().__init__()
        self.V_ref = torch.from_numpy(V_ref).float()
        self.all_instances = [
            Vertices(F=F, path_obj=path_obj, instance_idx=idx, V_ref=V_ref, num_samples=num_samples, landmarks=landmarks)
            for idx, path_obj in enumerate(paths_obj)]
        self.num_instances = len(self.all_instances)
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        ground_truth = [{
            'coords': obj['coords'] ,
            'positions': obj['positions'] ,
            'weight': obj['weight'] ,
            } for obj in observations]

        return observations, ground_truth


"""
Functions for accesssing 3DCaricShop dataset
"""
class CaricShop3D:
    """
    3DCaricShopDataset. The authors' registration with NICP to fit
    Gives access to (Name, Picture path, Vertex Position) tuples
    """
    NAMES_MASK = ['Contour', 'Contour2', 'Ears', 'Eyebrows', 'Eyes', 'Mouth', 'Nose']

    NAMES_LOCALITY = ['Ears', 'Eyebrows', 'Eyes', 'Mouth', 'Nose']

    def _build_map(self):
        """
        Build map: path --> index in the dataset
        """
        result = {}
        path_db = './sort_info.txt'
        with open(path_db, 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            tokens = line.split()
            key = ' '.join(tokens[:-2])
            value = int(tokens[-1])

            result[key] = value
        
        self.map = result

    def _convert_path_to_key(self, path):
        sp = path.split('/')
        leaf = sp[-1]
        subject = sp[-2]
        
        return '/'.join([subject, leaf])

    def __init__(self, dir, skip=False, path_pickle=None):
        path_template = os.path.join('staticdata', 'labelled_tMesh.obj')
        V_ref, F_ref, _, _, _, _ = read_obj(path_template)

        # Template mesh vertex position
        self.V_ref = V_ref
        # triangle indices shared between meshes
        self.F = F_ref
        
        self._build_map()
        self.paths_obj = []
        if not skip:
            dir_processed = os.path.join(dir, 'processedData', 'tMesh')
            for root, dirs, files in os.walk(dir_processed):
                for filename in files:
                    if filename.endswith('.obj'):
                        path_obj = os.path.join(root, filename)
                        self.paths_obj.append(path_obj)

        self.paths_obj = sorted(
            self.paths_obj, 
            key=lambda x:self.map[self._convert_path_to_key(x)])

        # Faces without the back of the head
        self.F_disk = rawlandmarks.CaricShop3D.TRIANGLES_SLICED.copy()
        # Mapping from original V_ref to the new sliced vertices
        self.J = rawlandmarks.CaricShop3D.MAP_ORIG_TO_SLICED.copy()
        # Texture coordinates. ranges of each axis: [-1, 1]. Defined on the sliced mesh.
        self.VT = rawlandmarks.CaricShop3D.VT.copy()

        # landmark vertex indices
        self.landmarks_disk = []
        raw = rawlandmarks.CaricShop3D.LANDMARKS.copy()
        for i in range(len(raw)):
            lmk = raw[i]
            pos = self.V_ref[lmk]
            dists = np.linalg.norm(self.V_ref[self.J] - pos, axis=1)
            newlmk = np.argmin(dists)

            self.landmarks_disk.append(newlmk)
        self.landmarks_disk = np.array(self.landmarks_disk, dtype=np.int32)
        self.landmarks = np.array(rawlandmarks.CaricShop3D.LANDMARKS, dtype=np.int32)
        self.landmarks_dlib = self.landmarks[rawlandmarks.CaricShop3D.DLIB_68]

        def get_all_neighbors(F, inds):
            this = set(inds)
            neighbors = set()
            li = igl.adjacency_list(F)
            for i in inds:
                local_neighbors = li[i]
                for j in local_neighbors:
                    neighbors.add(j)
                    
            return list(neighbors - this)
        self.boundary = igl.boundary_loop(self.F_disk)
        self.boudnary2 = np.array(get_all_neighbors(self.F_disk, self.boundary.tolist()), dtype=np.int32)
        self.boundary_tworing = np.concatenate([self.boundary, self.boudnary2]).astype(np.int32)

        self.landmarks_region_inds = {
            'Eyebrow_Right': rawlandmarks.CaricShop3D.DLIB_INDS_EYEBROW_RIGHT,
            'Eyebrow_Left': rawlandmarks.CaricShop3D.DLIB_INDS_EYEBROW_LEFT,

            'Eye_Right': rawlandmarks.CaricShop3D.DLIB_INDS_EYE_RIGHT,
            'Eye_Left': rawlandmarks.CaricShop3D.DLIB_INDS_EYE_LEFT,

            'Nose': rawlandmarks.CaricShop3D.DLIB_INDS_NOSE_FLAT,
            'Nose_Full': rawlandmarks.CaricShop3D.DLIB_INDS_NOSE,

            'Mouth': rawlandmarks.CaricShop3D.DLIB_INDS_MOUTH_FLAT,
            'Forehead': rawlandmarks.CaricShop3D.DLIB_INDS_FOREHEAD,
            'Chin': rawlandmarks.CaricShop3D.DLIB_INDS_CHIN,

            'Outer': rawlandmarks.CaricShop3D.DLIB_INDS_CHIN,

            'Cheekbone': rawlandmarks.CaricShop3D.INDS_CHEEKBONE,
            'Temple': rawlandmarks.CaricShop3D.INDS_TEMPLE,
            'Chin_Sparse': np.array([7, 9], dtype=np.int),

            'Sparse': np.array([18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 36, 49, 52, 55, 58], dtype=np.int) - 1,
        }

        self.landmarks_region = { k: self.landmarks[v] for k, v in self.landmarks_region_inds.items() }

        fixed = np.concatenate([self.boundary_tworing, self.landmarks_region['Cheekbone']])
        V = self.V_ref[self.J]
        F = self.F_disk
        self.V_flat = igl.harmonic_weights(V, F, fixed, V[fixed], 2)

        path_horlines = './staticdata/horlines.pkl'
        if os.path.exists(path_horlines):
            with open(path_horlines, 'rb') as fin:
                self.horlines = pickle.load(fin)
        else:
            self.horlines = None

        path_horlines = './staticdata/horlines_81.pkl'
        if os.path.exists(path_horlines):
            with open(path_horlines, 'rb') as fin:
                self.horlines_81 = pickle.load(fin)
        else:
            self.horlines_81 = None
            
    def __len__(self):
        return len(self.paths_obj)

    def __getitem__(self, i):
        return self.paths_obj[i]


class CaricShop3DTrain(Dataset):
    def __init__(self, dir, num_train=1268, num_samples=11581, use_landmarks=False):
        self.caricshop = CaricShop3D(dir)
        if use_landmarks:
            lmks = self.caricshop.landmarks
        else:
            lmks = None
        self.ds = VerticesMulti(self.caricshop.paths_obj[:num_train], self.caricshop.V_ref, self.caricshop.F, num_samples=num_samples, landmarks=lmks)

    def __getitem__(self, i):
        return self.ds[i]

    def __len__(self):
        return len(self.ds)

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

class PairWrapper(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self.ds[i], self.ds[i]

    def __len__(self):
        return len(self.ds)

class Caricature_data(Dataset):
    """
    Alive caricature 2D dataset
    """
    def __init__(self, dir_caricshop, dir_caricature_data):
        self.caricshop = CaricShop3D(dir_caricshop)
        dir_data = os.path.join(dir_caricature_data, 'Caricature_w_landmark')
        dir_caricature_crop = os.path.join(dir_caricature_data, 'my_result_Crop')

        self.ds = []
        for i in range(1, 51):
            path_img = os.path.join(dir_data, '{}.jpg'.format(i))
            path_lmk = os.path.join(dir_data, '{}.txt'.format(i))
            path_mesh = os.path.join(dir_caricature_crop, '{}.obj'.format(i))
            img = sio.imread(path_img)
            lmk = np.loadtxt(path_lmk)
            V, F = igl.read_triangle_mesh(path_mesh)
            lmk_flip = lmk.copy()
            height = img.shape[0]
            lmk_flip[:,1] = height - lmk_flip[:,1]
            self.ds.append((img, lmk, lmk_flip, V, F))

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self.ds[i]

    def __len__(self):
        return len(self.ds)

class Landmarkonly_data(Dataset):

    def __init__(self, dir_caricshop, dir_landmark_data):
        self.caricshop = CaricShop3D(dir_caricshop, skip=True)
        dir_data = dir_landmark_data

        self.ds = []
        path_lmks = sorted(glob(os.path.join(dir_data, '*.txt')))

        for path_lmk in path_lmks:
            lmk = np.loadtxt(path_lmk)
            lmk_flip = lmk.copy()
            lmk_flip[:,1] = -lmk_flip[:,1]
            self.ds.append((lmk, lmk_flip))

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self.ds[i]

    def __len__(self):
        return len(self.ds)

class CaricShop3DTestLandmark_Attr(Dataset):
    def __init__(self, dir, num_train=1268, dims=3):
        self.caricshop = CaricShop3D(dir)
        self.paths = self.caricshop.paths_obj[num_train:]
        self.dims = dims

    def __getitem__(self, i):
        return self.paths[i], Landmarks(
            self.paths[i], 0, self.caricshop.V_ref, self.caricshop.F, 
            inds_lmk=self.caricshop.landmarks, dims=self.dims)

    def __len__(self):
        return len(self.paths)
