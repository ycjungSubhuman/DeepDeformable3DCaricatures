import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os

from surface_deformation import create_mesh, create_mesh_single

def train(
    model, train_dataloader, epochs, lr, steps_til_summary, 
    epochs_til_checkpoint, model_dir, summary_dir, 
    loss_schedules=None, is_train=True, V_ref=None, F=None, landmarks=None, dims_lmk=3, **kwargs):

    print('Training Info:')
    print('batch_size:\t\t',kwargs['batch_size'])
    print('epochs:\t\t\t',epochs)
    print('len_dataloader:\t\t\t',len(train_dataloader))
    print('learning rate:\t\t',lr)
    for key in kwargs:
        if 'loss' in key:
            print(key+':\t',kwargs[key])
    
    if is_train:
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
    else:
        embedding = torch.zeros(128).float().cuda()
        embedding.requires_grad = True
        optim = torch.optim.Adam(lr=lr, params=[embedding])

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = summary_dir
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    mesh_dir = os.path.join(model_dir, 'meshes')
    utils.cond_mkdir(mesh_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint:
                if is_train:
                    torch.save(model.module.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                    # Save intermediate visualizations
                    for subject_idx_vis in range(128):
                        create_mesh(
                            model, 
                            os.path.join(mesh_dir, '{:04d}.obj'.format(subject_idx_vis)),
                            train_dataloader.dataset.ds.V_ref,
                            train_dataloader.dataset.caricshop.F,
                            embedding=model.module.latent_codes(torch.Tensor([subject_idx_vis]).long().cuda()))
                else:
                    embed_save = embedding.detach().squeeze().cpu().numpy()
                    np.savetxt(os.path.join(checkpoints_dir, 'embedding_epoch_%04d.txt' % epoch),
                               embed_save)

                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if is_train:
                    losses = model(model_input,gt)
                else:
                    losses = model.embedding(embedding, model_input,gt, landmarks, dims_lmk)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    if is_train:
                        torch.save(model.module.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1

        if is_train:
            torch.save(model.module.cpu().state_dict(),
                       os.path.join(checkpoints_dir, 'model_final.pth'))
        else:
            embed_save = embedding.detach().squeeze().cpu().numpy()
            np.savetxt(os.path.join(checkpoints_dir, 'embedding_epoch_%04d.txt' % epoch),
                       embed_save)
            scales = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

            for scale in scales:
                create_mesh_single(
                    model, 
                    os.path.join(checkpoints_dir, 'mesh_{}.obj'.format(scale)),
                    V_ref,
                    F,
                    embedding=embedding*scale)

        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
