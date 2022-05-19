import torch 
import numpy as np 
import pandas as pd
from tqdm import tqdm


def encoder_train(encoder, data_loader, optimizer, encoder_loss,linear_list,classifier_loss,optimizer_decoder,  device='cuda:0', scaler=None, use_amp=False, split="Valid", wandb=None, dataset=None,treshold=85, modality_contrastive=False):
    
    samples = 0.
    batch_count =0
    batch_count_linear = 0

    label_quality = 0
    n_pseudo_labels = 0 

    cumulative_contr_loss = 0.
    cumulative_ce_loss = 0.


    for _ , batch in enumerate(tqdm(data_loader)):
        
        targets = batch[0].to(device,non_blocking=True)
        features_s_1 =  batch[1]
        features_s_2 =  batch[2]
        sep_mod_pred = [torch.Tensor([]) for _ in range(len(linear_list)-1)]

        for feat_id, _ in enumerate(features_s_1):
            if len(features_s_1[feat_id]) != 0:
                features_s_1[feat_id] = features_s_1[feat_id].to(device,non_blocking=True).float()

        for feat_id, _ in enumerate(features_s_2):
            if len(features_s_2[feat_id]) != 0:
                features_s_2[feat_id] = features_s_2[feat_id].to(device,non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=use_amp):
            proj_feat_1, l_feat_1 = encoder(features_s_1)
            proj_feat_2, l_feat_2 = encoder(features_s_2)
        
        l_contr_feat = []
        
        if modality_contrastive:
            for i in range(len(proj_feat_1)):
                proj_1 = torch.nn.functional.normalize(proj_feat_1[i], dim=-1).unsqueeze(1)
                proj_2 = torch.nn.functional.normalize(proj_feat_2[i], dim=-1).unsqueeze(1)

                contr_feat = torch.cat((proj_1, proj_2), 1)
                l_contr_feat.append(contr_feat)
        else:
            # creates two parallel tensors that are the two view of the modality concatenated 
            contr_feat_1 = torch.Tensor([]).to(device=device)
            contr_feat_2 = torch.Tensor([]).to(device=device)
            for i in range(len(proj_feat_1)):
                contr_feat_1 = torch.cat((contr_feat_1,proj_feat_1[i]), 1)
                contr_feat_2 = torch.cat((contr_feat_2,proj_feat_2[i]), 1)
            
            contr_feat_1 = torch.nn.functional.normalize(contr_feat_1, dim=-1)
            contr_feat_2 = torch.nn.functional.normalize(contr_feat_2, dim=-1)
            
            contr_feat =  torch.cat((contr_feat_1.unsqueeze(1), contr_feat_2.unsqueeze(1)), 1)
            l_contr_feat = [contr_feat, contr_feat]

        n_losses = 0
        contr_loss = 0
        for idx_feat,feat in enumerate(l_contr_feat):
            for idx_innner,feat_inner in enumerate(l_contr_feat):
                if idx_feat != idx_innner:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        contr_loss += encoder_loss(feat, feat_inner)
                    n_losses += 1                            
        contr_loss /= n_losses
        
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(contr_loss).backward()
            scaler.step(optimizer)
        else:
            contr_loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        samples+=batch_size
        batch_count +=1

        if use_amp:
            scaler.update()
        
        cumulative_contr_loss += contr_loss.item() # Note: the .item() is needed to extract scalars from tensors

    final_contr_loss = cumulative_contr_loss/batch_count
    final_ce_loss = 0
    if label_quality > 0:
        final_ce_loss = cumulative_ce_loss/batch_count_linear
        label_quality = label_quality/batch_count_linear



    print(f"label_quality {label_quality} n_pseudo_labels {n_pseudo_labels}")
    if wandb is not None:
        wandb.log({""+split+"_train Contrastive Loss": final_contr_loss})
        wandb.log({""+split+"_train Cross Entropy Loss": final_ce_loss})
        wandb.log({""+split+"_train Pseudo label quality": label_quality})
        wandb.log({""+split+"_train n_pseudo_labels": n_pseudo_labels})
        

    