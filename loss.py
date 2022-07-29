import torch
import torch.nn.functional as F

def surface_deformation_pos_loss(model_output, gt):
    gt_pos = gt['positions']
    V_new = model_output['model_out']

    embeddings = model_output['latent_vec']

    data_constraint = torch.nn.functional.mse_loss(V_new, gt_pos)
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {'data_constraint': data_constraint * 3e3, 
            'embeddings_constraint': embeddings_constraint.mean() * 1e6}
