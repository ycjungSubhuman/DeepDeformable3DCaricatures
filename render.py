"""
Obtain video that demainstrates the training result
"""
import igl
import tqdm
import trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

from scipy.spatial.transform import Rotation

def get_colors(input, vmin, vmax, colormap="viridis"):
    colormap = plt.cm.get_cmap(colormap)
    norm = plt.Normalize(vmin, vmax, clip=True)
    return colormap(norm(input))[:, :3]

class MeshRenderer:

    def __init__(self):
        self.renderer = pyrender.OffscreenRenderer(512, 512)

    def render_mesh(self, V, F):
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]), 
            metallicFactor=0.0, roughnessFactor=0.4)
        tm = trimesh.Trimesh(V, F)
        mesh = pyrender.Mesh.from_trimesh(tm)
        
        scene = pyrender.Scene(ambient_light=0.00*np.ones(3))
        cam = pyrender.OrthographicCamera(1.00, 1.00)
        pose = np.array([
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 1.5, 2.0],
            [0.0, 0.0, 0.0, 1.0]])
        scene.add(mesh)
        scene.add(cam, pose=pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=pose)

        color, _ = self.renderer.render(scene)

        return color

    def render_mesh_tex(self, V, F, VT, texture):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.4,
            emissiveTexture=texture)
        tm = trimesh.Trimesh(V, F, process=False)
        tm.visual = trimesh.visual.TextureVisuals(uv=VT, image=texture)
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
        
        scene = pyrender.Scene(ambient_light=0.00*np.ones(3))
        cam = pyrender.OrthographicCamera(1.00, 1.00)
        pose = np.array([
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 1.5, 2.0],
            [0.0, 0.0, 0.0, 1.0]])
        scene.add(mesh)
        scene.add(cam, pose=pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=30.0)
        scene.add(light, pose=pose)

        color, _ = self.renderer.render(scene)

        return color
