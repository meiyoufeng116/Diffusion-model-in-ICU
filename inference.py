# from curses import tparm
import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader,Dataset

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

from imputers.Diffmodel import Diffmodel

from sklearn.metrics import mean_squared_error,roc_curve, auc,mean_absolute_error,roc_auc_score

from statistics import mean
# from dtaidistance import dtw

import matplotlib.pyplot as plt

import pandas as pd


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

def normalize(a):

    return (a-a.mean())/(a.max()-a.min()+0.00001)


class mimic_datasetVT(Dataset):
    def __init__(self,data,gt):
        self.data=data
        self.gt=gt
    def __len__(self):
        return len(self.data)
    def __getitem__(self,key):
        return self.data[key],self.gt[key]

def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
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

    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)

        
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
        
    ### Custom data loading and reshaping ###
    
    mu, sigma, samples, alarm, rule_based_results, groundtruth = torch.load(
        'datasets/standardized_samples_alarms_results_embedding_labels.pt')
    # mu, sigma, samples, alarm, rule_based_results, groundtruth = torch.load(
    #     'datasets/2015_phy_stanardized.pt')


    VT_index,VT_True_index,VT_False_index=[],[],[]
    VIFB_index,VIFB_F_index=[],[]
    for i in range(len(samples)):
        if alarm[i][0]==1 :
            VT_index.append(i)
        if alarm[i][0]==0:
            VIFB_index.append(i)
    # for i in VT_index[-670:]:
    #     if alarm[i][0]==1 and groundtruth[i]==0:
    #         VT_True_index.append(i)
    #     if alarm[i][0]==1 and groundtruth[i]==1:
    #         VT_False_index.append(i)

    VT_samples= samples[VT_index]
    VIFB_samples=samples[VIFB_index]

    # VT_samples[:,8,:]=VT_samples[:,8,:]*0.1
    # observed_values = VT_samples[:,:,::2][:,:12,37000:37500]*0.1
    
    observed_values = VT_samples[-1000:,:,37000:37500]
    test_gt=groundtruth[VT_index][-1000:]
    
    # observed_values = VT_samples[:,:,::2][:,:12,37000:37500]
    # observed_values[:,8,:] = observed_values[:,8,:]*0.1
    
    for i in range(12):
        for j in range(observed_values.shape[0]):
            observed_values[j,i,:]=normalize(observed_values[j,i,:])
    
    # test_gt=groundtruth[VT_index][:]
    
    # observed_values = VIFB_samples[-100:,:,37250:37750]*0.1     #VFB
    # test_gt=groundtruth[VIFB_index][-100:]
    # test_values=VT_samples[-670:,8:9,37000:38000].permute(0,2,1)




    test_set=mimic_datasetVT(observed_values,test_gt)
    testing_data=DataLoader(test_set,batch_size=200)
    
    
    
    
    
    
    # testing_data = np.load(trainset_config['test_data_path'])
    # testing_data = np.split(testing_data, 4, 0)
    # testing_data = np.array(testing_data)
    # testing_data = torch.from_numpy(testing_data).float().cuda()
    print('Data loaded')

    all_mse,t_mse,f_mse = [],[],[]
    mse_loss=nn.MSELoss()
    mae_loss=nn.L1Loss()
    

    dists,dists_a=[],[]
    generated_audios=np.empty([1,12,500])
    origin=observed_values.numpy()
    gt=test_gt.numpy()
    
    outfile = f'gt{i}.npy'
    new_out = os.path.join(ckpt_path, outfile)
    np.save(new_out, gt)
    
    outfile = f'original{i}.npy'
    new_out = os.path.join(ckpt_path, outfile)
    np.save(new_out, origin)


    
    dtws=[]
    for i, batch in enumerate(testing_data):
        gt=batch[1]
        gt_true=np.argwhere(gt.numpy()==1)[:,0]
        gt_false=np.argwhere(gt.numpy()==0)[:,0]
        batch=batch[0].cuda()
        if masking == 'mnr':
            mask_T = get_mask_mnr(batch[0].permute(1,0), missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'rm':
            mask_T = get_mask_rm(batch[0].permute(1,0), missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()


        # mask = mask.permute(0,2,1)
        # batch = batch.permute(0,2,1)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        sample_length = batch.size(2)
        sample_channels = batch.size(1)
        num_samples = batch.size(0)
        generated_audio = sampling(net, (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)


        for i in range(batch.size(0)):
            dist_dtws=0.0
            dist=mse_loss((generated_audio*~mask.bool())[i,:,:],(batch*~mask.bool())[i,:,:])
            dist_a=mae_loss((generated_audio*~mask.bool())[i,:,:],(batch*~mask.bool())[i,:,:])
            # dist=mse_loss((generated_audio*~mask.bool())[i,:,:],(torch.zeros(batch.shape).cuda()*~mask.bool())[i,:,:])
            # for j in range(12):
            #     dist_dtw=dtw.distance((generated_audio*~mask.bool())[i,j,:],(batch*~mask.bool())[i,j,:])
            #     dist_dtws=dist_dtw+dist_dtws
            # dtws.append(dist_dtws)
            dists.append(dist.detach().cpu())
            dists_a.append(dist_a.detach().cpu())
        
        
        
        end.record()
        torch.cuda.synchronize()

        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                                             ckpt_iter,
                                                                                             int(start.elapsed_time(
                                                                                                 end) / 1000)))

        
        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy() 
        
        generated_audios=np.concatenate((generated_audios,generated_audio),axis=0)
        # outfile = f'imputation{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, generated_audio)

        # outfile = f'original{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, batch)

        # outfile = f'mask{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, mask)

        print('saved generated samples at iteration %s' % ckpt_iter)
        # mask = mask.repeat(num_samples,1)
        mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        if gt_false.size!=0 and gt_true.size!=0:
            mse_t=mean_absolute_error(generated_audio[gt_true][~mask[gt_true].astype(bool)], batch[gt_true][~mask[gt_true].astype(bool)])
            mse_f=mean_absolute_error(generated_audio[gt_false][~mask[gt_false].astype(bool)], batch[gt_false][~mask[gt_false].astype(bool)])
            t_mse.append(mse_t)
            f_mse.append(mse_f) 
        all_mse.append(mse)


    outfile = f'imputation{i}.npy'
    new_out = os.path.join(ckpt_path, outfile)
    np.save(new_out, generated_audios)
    
    outfile = f'mask{i}.npy'
    new_out = os.path.join(ckpt_path, outfile)
    np.save(new_out, mask)
    
    tpr,tfr,ths=roc_curve(test_gt,dists)

    plt.plot(tfr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig("roc.jpg")

    plt.clf()   
    
      
    Score=[0]
    Max_THRESHOLD={}

    for t in ths:
        THRESHOLD = t # Anomaly score threshold for an instance to be considered as anomaly 

        #TIME_STEPS = dataset.window_length
        test_score_df = pd.DataFrame(index=range(test_gt.shape[0]))
        test_score_df['loss'] = dists
        test_score_df['y'] = test_gt.numpy()
        test_score_df['threshold'] = THRESHOLD
        test_score_df['anomaly'] = test_score_df.loss < test_score_df.threshold
        # test_score_df['t'] = [x[59].item() for x in test_dataset.x]

        aUC=roc_auc_score(test_score_df['y'] ,test_score_df['anomaly'])

        start_end = []
        state = 0
        # for idx in test_score_df.index:
        #     if state==0 and test_score_df.loc[idx, 'y']==1:
        #         state=1
        #         start = idx
        #     if state==1 and test_score_df.loc[idx, 'y']==0:
        #         state = 0
        #         end = idx
        #         start_end.append((start, end))

        # for s_e in start_end:
        #     if sum(test_score_df[s_e[0]:s_e[1]+1]['anomaly'])>0:
        #         for i in range(s_e[0], s_e[1]+1):
        #             test_score_df.loc[i, 'anomaly'] = 1
        # test_score_df.to_csv("test.csv")
        actual = np.array(test_score_df['y'])
        predicted = np.array([int(a) for a in test_score_df['anomaly']])

        predicted = np.array(predicted)
        actual = np.array(actual)

        tp = np.count_nonzero(predicted * actual)
        tn = np.count_nonzero((predicted - 1) * (actual - 1))
        fp = np.count_nonzero(predicted * (actual - 1))
        fn = np.count_nonzero((predicted - 1) * actual)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        score=(tp+tn)/(tp+tn+fp+5*fn)
        if score>= max(Score):
            Score.append(score)
            Max_THRESHOLD.clear()
            Max_THRESHOLD["thershold"]=t
            Max_THRESHOLD["TP"]=tp
            Max_THRESHOLD["TN"]=tn
            Max_THRESHOLD["FP"]=fp
            Max_THRESHOLD["FN"]=fn
            Max_THRESHOLD["accuracy"]=accuracy
            Max_THRESHOLD["score"]=score
            Max_THRESHOLD["TPR"]=tp/(tp+fn)
            Max_THRESHOLD["TNR"]=tn/(tn+fp)

    plt.plot(test_score_df.index, test_score_df.loss, label='loss')
    # plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    plt.axhline(Max_THRESHOLD["thershold"], label='threshold',color='r')
    plt.plot(test_score_df.index, test_gt.numpy(), label='y')
    plt.xticks(rotation=25)
    plt.legend()
    plt.savefig(str(12)+"pic.png",dpi=250)
    plt.clf()
    print("acc:",accuracy,"score:",score,"tpr:",tp/(tp+fn),"tnr:",tn/(tn+fp))
    print(Max_THRESHOLD)
    print("AUC:",auc(tfr,tpr))

    
    print('Total MSE:', mean(all_mse))
    print('Total tMSE:', mean(t_mse))
    print('Total fMSE:', mean(f_mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSDS4.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=50,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

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

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             use_model=train_config["use_model"],
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"])
