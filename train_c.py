import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams,training_feature
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm

from imputers.Diffmodel import Diffmodel

from InfoNCE import *

# set random seed
SEED = 1000
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

from torch.utils.tensorboard import SummaryWriter



def normalize(a):
    if a[0]!=0:
        return (a-a.mean())/(a.max()-a.min()+0.00001)
    else:
        return(a)

class mimic_datasetVT(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,key):
        return self.data[key]


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k,
          dataset,
          dataset_path):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    if use_model == 0:
        net = Diffmodel(**model_config).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)
    # net = nn.DataParallel(net)
    # net.cuda()

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

        
        
    
    ### Custom data loading and reshaping ###
        
    mu, sigma, samples, alarm, rule_based_results, groundtruth = torch.load(dataset_path)
    # mu, sigma, samples, alarm, rule_based_results, groundtruth = torch.load(
    #     'datasets/2015_phy_stanardized.pt')


    VT_index=[]
    VT_F_index=[]
    VFIB_index,VFIB_F_index=[],[]
    for i in range(len(samples)):
        if alarm[i][0]==1 and groundtruth[i]==1:
            VT_index.append(i)
        if alarm[i][0]==1 and groundtruth[i]==0:
            VT_F_index.append(i)  
        if alarm[i][1]==1 and groundtruth[i]==1:
            VFIB_index.append(i)
        if alarm[i][1]==1 and groundtruth[i]==0:
            VFIB_F_index.append(i)
            

    VT_samples= samples[VT_index]
    VT_F_samples=samples[VT_F_index]
    VFIB_samples=samples[VFIB_index]
    VFIB_F_samples=samples[VFIB_F_index]

    if dataset=='VT':
        observed_values = VT_samples[:1000,:,37000:37500]               #VT
        observed_values_f =VT_F_samples[:1000,:,37000:37500]
        gt_t=groundtruth[VT_index][:1000]

    if dataset=='VFIB':
        observed_values = VFIB_samples[:100,:,37200:37700]            #V-FIB
        observed_values_f =VFIB_F_samples[:250,:,37200:37700]

    training_set_T=mimic_datasetVT(observed_values)
    training_set_F=mimic_datasetVT(observed_values_f)
    training_data=DataLoader(training_set_T,batch_size=1)
    training_data_f=DataLoader(training_set_F,batch_size=32)
    infonce=InfoNCE(negative_mode='unpaired')

    print('Data loaded')
    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch,batch_f in zip(training_data,training_data_f):
            
            batch,batch_f=batch.cuda(),batch_f.cuda()
                
            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0].permute(1,0), missing_k)
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0].permute(1,0), missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)

            mask = transposed_mask.permute(1, 0)
            # mask = mask.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            mask_f=transposed_mask.permute(1, 0).repeat(batch_f.size()[0], 1, 1).float().cuda()
            for i in range(batch.shape[0]):
                for j in range(12):
                    if batch[i,j,10]==0:
                        mask[i,j,:]=0
                    if batch_f[i,j,10]==0:
                        mask_f[i,j,:]=0
            
            mask_f=transposed_mask.permute(1, 0).repeat(batch_f.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            loss_mask_f=~mask_f.bool()


            assert batch.size() == mask.size() == loss_mask.size()

            #batch size, channels, length
            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            Y = batch_f,batch_f,mask_f,loss_mask_f
            loss,feature = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)
            feature_t = training_feature(net, Y, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)
            loss_c=infonce(feature.view(batch.shape[0],-1),feature.view(batch.shape[0],-1),feature_t.view(batch_f.shape[0],-1))
            loss_t=loss+0*loss_c
            loss_t.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}\tloss_t:{}".format(n_iter, loss.item(),loss_t.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    train(**train_config)
