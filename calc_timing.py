import torch
import time

from configs import Configs
from utils.plotting import get_projections, get_plots, MAP, offset, layer_bottom_pos, cell_thickness, Xmax, Xmin, Zmax, Zmin
from tqdm import tqdm

import numpy as np
import torch.nn as nn

from pyro.nn import ConditionalDenseNN, DenseNN, ConditionalAutoRegressiveNN
import pyro.distributions as dist
import pyro.distributions.transforms as T
from custom_pyro import ConditionalAffineCouplingTanH

from models.CaloClouds import *


########################
### PARAMS #############
########################

cfg = Configs()
cfg.device = 'cpu'
#use single thread
torch.set_num_threads(1)

# min and max energy of the generated events
min_e = 10
max_e = 100

num = 2000 # total number of generated events

bs = 1 # batch size   # optimized: bs=64 for GPU, bs=512 for CPU (multi-threaded), bs=1 for CPU (single-threaded)

iterations = 5 # number of iterations for timing


########################
########################
########################


def main(cfg, min_e, max_e, num, bs, iterations):

    num_blocks = 10
    flow, distribution = compile_HybridTanH_model(num_blocks, 
                                            num_inputs=32, ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than 
                                            num_cond_inputs=1, device=cfg.device)  # num_cond_inputs

    checkpoint = torch.load('/beegfs/desy/user/akorol/chekpoints/ECFlow/EFlow+CFlow_138.pth')
    flow.load_state_dict(checkpoint['model'])
    flow.eval().to(cfg.device)


    cfg.sched_mode = 'quardatic'
    model = CaloClouds(cfg).to(cfg.device)

    checkpoint = torch.load('./logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s_MSE_loss_smired_possitions_quardatic2023_04_06__16_34_39/ckpt_0.000000_837000.pt') # quadratic


    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    times_per_shower = []
    for _ in range(iterations):

        s_t = time.time()
        fake_showers = gen_showers_batch(distribution, model, min_e, max_e, num, bs)
        t = time.time() - s_t
        print(fake_showers.shape)
        print('total time (seconds): ', t)
        print('time per shower (seconds): ', t / num)
        times_per_shower.append(t / num)

    print('mean time per shower (seconds): ', np.mean(times_per_shower))
    print('std time per shower (seconds): ', np.std(times_per_shower))




def compile_HybridTanH_model(num_blocks, num_inputs, num_cond_inputs, device):
    # the latent space distribution: choosing a 2-dim Gaussian
    base_dist = dist.Normal(torch.zeros(num_inputs).to(device), torch.ones(num_inputs).to(device))

    input_dim = num_inputs
    count_bins = 8
    transforms = []
    transforms2 = []
      
    input_dim = num_inputs
    split_dim = num_inputs//2
    param_dims1 = [input_dim-split_dim, input_dim-split_dim]
    param_dims2 = [input_dim * count_bins, input_dim * count_bins, input_dim * (count_bins - 1), input_dim * count_bins]

    torch.manual_seed(42)

    for _ in range(num_blocks):
        

                    
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)

        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        
        
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        hypernet = DenseNN(num_cond_inputs, [input_dim*4, input_dim*4], param_dims2)
        ctf = T.ConditionalSpline(hypernet, input_dim, count_bins)
        transforms2.append(ctf)
        transforms.append(ctf)
        

        

        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)

        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)

        
        hypernet = ConditionalDenseNN(split_dim, num_cond_inputs, [input_dim*10, input_dim*10], param_dims1)
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet)
        transforms2.append(ctf)
        transforms.append(ctf)
        
        perm = torch.randperm(input_dim, dtype=torch.long).to(device)
        ff = T.Permute(perm)
        transforms.append(ff)
        
        
    modules = nn.ModuleList(transforms2)

    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transforms)

    return modules, flow_dist



def get_scale_factor(num_clusters):
    
    coef_real = np.array([ 2.57988645e-09, -2.94056522e-05,  3.42194568e-01,  5.34968378e+01])
    coef_fake = np.array([ 3.85057207e-09, -4.16463897e-05,  4.19800713e-01,  5.82246858e+01])
    
    poly_fn_real = np.poly1d(coef_real)
    poly_fn_fake = np.poly1d(coef_fake) 
    
    scale_factor = poly_fn_real(num_clusters) / poly_fn_fake(num_clusters)

    return scale_factor



def get_shower(model, num_points, energy, cond_N, bs=1):

    e = torch.ones((bs, 1), device=cfg.device) * energy
    n = torch.ones((bs, 1), device=cfg.device) * cond_N
    
    if cfg.norm_cond:
        e = e / 100 * 2 -1   # max incident energy: 100 GeV
        n = n / cfg.max_points * 2  - 1
    cond_feats = torch.cat([e, n], -1)
        
    with torch.no_grad():
        fake_shower = model.sample(cond_feats, num_points, cfg.flexibility)
    
    return fake_shower



# batch inference 
def gen_showers_batch(distribution, model, e_min, e_max, num=2000, bs=32):
    
    cond_E = torch.FloatTensor(num, 1).uniform_(e_min, e_max).to(cfg.device)
    samples = distribution.condition(cond_E/100).sample(torch.Size([num, ])).cpu().numpy()
    
    energies = (samples[:, 0] * 2.5 * 1000).reshape(num, 1)
    
    clusters_per_layer_gen = samples[:, 2:] * 400
    clusters_per_layer_gen[clusters_per_layer_gen < 0] = 0

    scale_factor = get_scale_factor(clusters_per_layer_gen.sum(axis=1))
    scale_factor = np.expand_dims(scale_factor, axis=1)

    clusters_per_layer_gen = (clusters_per_layer_gen * scale_factor).astype(int)

    # sort by number of clusters for better batching and faster inference
    mask = np.argsort(clusters_per_layer_gen.sum(axis=1))
    clusters_per_layer_gen = clusters_per_layer_gen[mask]
    energies = energies[mask]
    cond_E = cond_E[mask]

    fake_showers_list = []
    
    
    for evt_id in tqdm(range(0, num, bs)):
        if (num - evt_id) < bs:
            bs = num - evt_id
        # convert clusters_per_layer_gen to a fractions of points in the layer out of sum(points in the layer) of event
        # multuply clusters_per_layer_gen by corrected tottal num of points
        hits_per_layer_all = clusters_per_layer_gen[evt_id : evt_id+bs] # shape (bs, num_layers) 
        max_num_clusters = hits_per_layer_all.sum(axis=1).max()
        cond_N = torch.Tensor(hits_per_layer_all.sum(axis=1)).to(cfg.device).unsqueeze(-1)
        
        fs = get_shower(model, max_num_clusters, cond_E[evt_id : evt_id+bs], cond_N, bs=bs)
        fs = fs.cpu().numpy()\
        
        fs[:, :, -1][fs[:, :, -1]  < 0] = 0    # setting negative energies to 0
        fs[:, :, -1] = fs[:, :, -1] / fs[:, :, -1].sum(axis=1).reshape(bs, 1) * energies[evt_id : evt_id+bs] # energy rescaling to predicted e_sum

        length = 6000 - fs.shape[1]
        fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)
        fake_showers_list.append(fs)
        
    fake_showers = np.vstack(fake_showers_list)

    # y calibration
    for i, hits_per_layer in enumerate(clusters_per_layer_gen):

        n_hits_to_concat = 6000 - hits_per_layer.sum()

        y_flow = np.repeat(layer_bottom_pos+cell_thickness/2, hits_per_layer)
        y_flow = np.concatenate([y_flow, np.zeros(n_hits_to_concat)])

        mask = np.concatenate([ np.ones( hits_per_layer.sum() ), np.zeros( n_hits_to_concat ) ])

        fake_showers[i, :, 1][mask == 0] = 10
        idx_dm = np.argsort(fake_showers[i, :, 1])
        fake_showers[i, :, 1][idx_dm] = y_flow


        fake_showers[i, :, :][y_flow==0] = 0 

    
    fake_showers = np.moveaxis(fake_showers, -1, -2)
    fake_showers[:, 0, :] = (fake_showers[:, 0, :] + 1) / 2
    fake_showers[:, 2, :] = (fake_showers[:, 2, :] + 1) / 2
    
    fake_showers[:, 0] = fake_showers[:, 0] * (Xmin-Xmax) + Xmax
    fake_showers[:, 2] = fake_showers[:, 2] * (Zmin-Zmax) + Zmax
    
    return fake_showers





if __name__ == '__main__':
    main(cfg, min_e, max_e, num, bs, iterations)