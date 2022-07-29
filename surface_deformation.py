import igl
import numpy as np
import time
import torch

def create_mesh(model, filename, V_ref, F, embedding=None):
    """
    From trained model and embeddings, create meshes
    """
    device = embedding.device
    coords = V_ref.to(device)
    V = model.module.inference(coords, embedding)
    V = V.cpu().numpy()[0]
    igl.write_obj(filename, V, F)

def create_mesh_single(model, filename, V_ref, F, embedding=None):
    """
    From trained model and embeddings, create meshes
    """
    device = embedding.device
    coords = V_ref.to(device)
    V = model.inference(coords, embedding)
    V = V.cpu().numpy()[0]
    igl.write_obj(filename, V, F)

