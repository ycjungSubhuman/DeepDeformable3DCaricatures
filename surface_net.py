import torch
from torch import nn
import modules
from meta_modules import HyperNetwork
from loss import *

class SurfaceDeformationField(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1, num_hidden_layers=3, hyper_hidden_features=256,hidden_num=128, **kwargs):
        super().__init__()

        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        # Deform-Net
        self.deform_net=modules.SingleBVPNet(type=model_type,mode='mlp', hidden_features=hidden_num, num_hidden_layers=num_hidden_layers, in_features=3,out_features=3)
        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.deform_net)
        print(self)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding

    # for generation
    def inference(self, coords, embedding):
        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)

            deformation = model_output['model_out']
            new_coords = coords + deformation
            return new_coords

    def inference_back(self, coords, embedding):
        model_in = {'coords': coords}
        hypo_params = self.hyper_net(embedding)
        model_output = self.deform_net(model_in, params=hypo_params)

        deformation = model_output['model_out']
        new_coords = coords + deformation
        return new_coords

    # for training
    def forward(self, model_input, gt):
        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates

        # get network weights for Deform-net using Hyper-net 
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)

        model_output = self.deform_net(model_input, params=hypo_params)
        displacement = model_output['model_out'].squeeze()
        V_new = coords + displacement # deform into template space

        model_out = {
            'model_in':model_output['model_in'], 
            'model_out':V_new, 
            'latent_vec':embedding, 
            'hypo_params':hypo_params}

        losses = surface_deformation_pos_loss(model_out, gt)
        return losses

    # for evaluation
    def embedding(self, embed, model_input, gt, landmarks=None, dims_lmk=3):
        coords = model_input['coords'] # 3 dimensional input coordinates
        embedding = embed
        hypo_params = self.hyper_net(embedding)

        model_output = self.deform_net(model_input, params=hypo_params)
        displacement = model_output['model_out'].squeeze()
        V_new = coords + displacement # deform into template space

        model_out = {
            'model_in':model_output['model_in'],
            'model_out':V_new[:,:dims_lmk],
            'latent_vec':embedding, 
            'hypo_params':hypo_params}
        gt['positions'] = gt['positions'][:,:dims_lmk]

        losses = surface_deformation_pos_loss(model_out, gt)
        return losses
