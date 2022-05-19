import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from statistics import mean


def evaluate_model(encoder, linear_list, data_loader, encoder_loss, classifier_loss,device='cuda:0', unsupervised=False, n_classes=7, dataset=None,scaler=None, split="Valid",wandb=None, label=None,use_amp=False):
    # perfomance evaluation on training set or evalutaion set, the flow is very similar to the training loop
    # with the exception that we are not training on it :)

    samples = 0.
    batch_count =0

    cmu_mosei_flag = dataset=="CMU_MOSEI" or dataset=="CMU_MOSEI_RAW"
    
    if cmu_mosei_flag:
        from utils.utils import weighted_acc as accuracy_score
    else:
        from sklearn.metrics import accuracy_score

    cumulative_loss = 0.
    cumulative_contr_loss = 0.
    cumulative_ce_loss = 0.

    sep_mod_pred = [torch.Tensor([]) for _ in range(len(linear_list)-1)]

    encoder.eval()
    for linear in linear_list:
        linear.eval()

    tot_target = torch.Tensor([])
    tot_prediction = torch.Tensor([])

    if cmu_mosei_flag:
        sigm = torch.nn.Sigmoid()
    
    skip_contrastive = False

    with torch.no_grad():
        for _ , batch in enumerate(tqdm(data_loader)):
            
            targets = batch[0].to(device,non_blocking=True)
            features_s_1 =  batch[1]
            features_s_2 =  batch[2]

            for feat_id, _ in enumerate(features_s_1):
                if len(features_s_1[feat_id]) != 0:
                    features_s_1[feat_id] = features_s_1[feat_id].to(device,non_blocking=True).float()

            for feat_id, _ in enumerate(features_s_2):
                if len(features_s_2[feat_id]) != 0:
                    features_s_2[feat_id] = features_s_2[feat_id].to(device,non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=use_amp):
                proj_feat_1, l_feat_1 = encoder(features_s_1)
                proj_feat_2, l_feat_2 = encoder(features_s_2)
          

            # loop over each modality 
            head_batch_pred = [None for _ in range(len(linear_list)-1)] 
            for feat_dim in range(len(l_feat_1)):
                feat_separate = l_feat_1[feat_dim].detach()
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = linear_list[feat_dim+1](feat_separate) # +1 because 0 is the combined
                
                if not cmu_mosei_flag:
                    _, logits = logits.max(1)
                else:
                    logits = sigm(logits).round()

                sep_mod_pred[feat_dim] = torch.cat((sep_mod_pred[feat_dim], logits.detach().cpu()), 0)
                head_batch_pred[feat_dim] = logits.detach()
            
            feat_1 =  torch.Tensor([]).to(device,non_blocking=True)
            feat_2 =  torch.Tensor([]).to(device,non_blocking=True)

            for i in range(len(l_feat_1)):
                feat_1 = torch.cat((feat_1,l_feat_1[i]), -1)
                feat_2 = torch.cat((feat_2,l_feat_2[i]), -1)

            l_contr_feat = []

            predicted_features = torch.Tensor([]).to(device,non_blocking=True)
            if encoder.prediction_head:
                predicted_modal_1 = proj_feat_1[-1].unsqueeze(-2) # [b_size, n_modality, n_modality-1, feat_dim]
                predicted_modal_2 =  proj_feat_2[-1].unsqueeze(-2)

                predicted_features =  torch.cat((predicted_modal_1,predicted_modal_2),-2)

                proj_feat_1 = proj_feat_1[:-1]
                proj_feat_2 = proj_feat_2[:-1]

            for i in range(len(proj_feat_1)):
                contr_feat = torch.cat((proj_feat_1[i].unsqueeze(1), proj_feat_2[i].unsqueeze(1)), 1)
                l_contr_feat.append(contr_feat)

            if not skip_contrastive:
                if unsupervised:
                    n_losses = 0
                    contr_loss = 0
                    for idx_feat,feat in enumerate(l_contr_feat):
                        for idx_innner,feat_inner in enumerate(l_contr_feat):
                            if idx_feat != idx_innner:
                                with torch.cuda.amp.autocast(enabled=use_amp):
                                    contr_loss += encoder_loss(feat, feat_inner)
                                n_losses += 1                            
                    contr_loss /= n_losses
                else:
                    n_losses = 0
                    contr_loss = 0
                    if encoder.prediction_head:
                        for idx_feat in range(predicted_features.shape[1]):
                            pred_head_idx = 0
                            for idx_innner,feat_inner in enumerate(l_contr_feat):
                                tgt = targets
                                if idx_feat != idx_innner:
                                    feat = predicted_features[:,idx_feat,pred_head_idx]
                                    with torch.cuda.amp.autocast(enabled=use_amp):
                                        contr_loss += encoder_loss(feat, feat_inner.detach(), tgt)
                                    pred_head_idx += 1
                                    n_losses += 1
                    else:
                        for idx_feat,feat in enumerate(l_contr_feat):
                            for idx_innner,feat_inner in enumerate(l_contr_feat):
                                tgt = targets
                                if idx_feat != idx_innner:
                                    with torch.cuda.amp.autocast(enabled=use_amp):
                                        contr_loss += encoder_loss(feat, feat_inner, tgt)
                                    n_losses += 1
                    contr_loss /= n_losses
            else:
                contr_loss = 0
               
            if not skip_contrastive:
                video_feat = feat_1.detach()
            else:
                video_feat = feat_1
               
            if cmu_mosei_flag:
                targets = targets[:, 1:]
                targets[targets>0] = 1
                targets = targets.float()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = linear_list[0](video_feat)
                ce_loss = classifier_loss(logits, targets)
            
            loss = contr_loss + ce_loss
            batch_size = targets.shape[0]
            samples+=batch_size
            batch_count +=1

            cumulative_loss += loss.item()
            if not skip_contrastive:
                cumulative_contr_loss += contr_loss.item() # Note: the .item() is needed to extract scalars from tensors
            else:
                cumulative_contr_loss += 0
            cumulative_ce_loss += ce_loss.item() # Note: the .item() is needed to extract scalars from tensors
            
            if not cmu_mosei_flag:
                _, logits = logits.max(1)
            else:
                logits =  sigm(logits).round()

            tot_prediction = torch.cat((tot_prediction,logits.detach().cpu()), 0)
            tot_target =  torch.cat((tot_target, targets.cpu()), 0)


    final_loss = cumulative_loss/batch_count
    final_contr_loss = cumulative_contr_loss/batch_count
    final_ce_loss = cumulative_ce_loss/batch_count

    if cmu_mosei_flag:
        tot_target = tot_target.numpy()
        tot_prediction =tot_prediction.numpy()
        #compute the accuracy and f1 score for each label 
        label_f1 = [] 
        label_accuracy = []  
        for idx in range(tot_target.shape[1]):
            f1 = f1_score(tot_target[:,idx], tot_prediction[:,idx],average='weighted')
            #f1 = f1_score(tot_target[:,idx], tot_prediction[:,idx])

            label_f1.append(f1)
            accuracy = 100 * accuracy_score(tot_target[:,idx], tot_prediction[:,idx])
            label_accuracy.append(accuracy)

        f1 = mean(label_f1)
        accuracy =  mean(label_accuracy) #100 *accuracy_score(tot_target, tot_prediction) #mean(label_accuracy)
        accuracy_sep_mod = [] 

        for mod in sep_mod_pred:
            mod = mod.numpy()
            acc_l_sep_mod = []
            for idx in range(tot_target.shape[1]):
                acc = 100 * accuracy_score(tot_target[:,idx], mod[:,idx])
                acc_l_sep_mod.append(acc)
            accuracy_sep_mod.append(mean(acc_l_sep_mod))
    else:
        tot_target = tot_target.numpy()
        tot_prediction =tot_prediction.numpy()

        #f1 = f1_score(tot_target, tot_prediction) 
        f1 = f1_score(tot_target, tot_prediction, average="weighted")
        label_f1 = [] # not used, we do not track single label f1 score for a single label
        matrix = confusion_matrix(tot_target, tot_prediction)
        label_accuracy = matrix.diagonal()/matrix.sum(axis=1)
        accuracy = 100 * accuracy_score(tot_target, tot_prediction)
        accuracy_sep_mod = [] 

        if wandb is not None:
            wandb.log({split+"_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                    y_true=tot_target, preds=tot_prediction,
                    class_names=label) })

        for mod in sep_mod_pred:
            accuracy_mod = 100* accuracy_score(tot_target, mod.numpy())
            accuracy_sep_mod.append(accuracy_mod)

    encoder.train()
    for linear in linear_list:
        linear.train()

    return final_loss, final_contr_loss, final_ce_loss, accuracy, label_accuracy, label_f1,accuracy_sep_mod, f1


def log_evaluation_performance(loss, contr_loss, ce_loss, accuracy, label_accuracy, label_f1, accuracy_separate, f1, set="valid",label=None, wandb= None, sep_acc_mod=None, dataset=None):

    cmu_mosei_flag = dataset=="CMU_MOSEI" or dataset=="CMU_MOSEI_RAW"
    print(f" {set} loss {round(loss,5)}, {set}_contr_loss {round(contr_loss,5)}, {set}_ce_loss {round(ce_loss,5)}, {set} accuracy {round(accuracy,2)}")
    if wandb is not None:
        wandb.log({""+set+"_Accuracy": accuracy , ""+set+"_Contrastive Loss": contr_loss, 
                ""+set+"_Cross Entropy Loss": ce_loss, ""+set+"_F1": f1, ""+set+"_Total Loss": loss})
        
        for idx, acc in enumerate(accuracy_separate):
            wandb.log({""+set+"_Accuracy_"+sep_acc_mod[idx]:acc})

        for i in range(len(label)):
            wandb.log({""+set+"_label_percentage_"+str(label[i]): label_accuracy[i]})
            if cmu_mosei_flag:
                wandb.log({""+set+"_label_f1_"+str(label[i]): label_f1[i]})