from math import log
import torch
from torch.autograd import backward
from utils.evaluate import evaluate_model
from utils.evaluate import log_evaluation_performance
from tqdm import tqdm
import os
from time import gmtime, strftime
from shutil import copyfile
import numpy as np
import yaml
from utils.ModelMonitor import ModelMonitoring
from utils.SupConSeparate import SupConLoss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from statistics import mean
from utils.utils import time_print
from utils.encoder_train import encoder_train
import copy


def train(encoder, linear_list, loader_train, optimizer, scheduler, encoder_loss, classifier_loss, wandb, epochs=200, device="cuda:2", test=False, loader_test=None, loader_valid=None, use_amp=False, dataset=None,  output_dir=None, config_file=None, cmu_mosei_flag=False, confidence=False, combine=False, modality_contrastive=True):

    # since MOSEI is a multilabel and is unbalanced as they do in literature
    # we compute the weighted accuracy
    if dataset == "CMU_MOSEI" or dataset == "CMU_MOSEI_RAW":
        from utils.utils import weighted_acc as accuracy_score
    else:
        from sklearn.metrics import accuracy_score

    # create the directory where the checkpoint and the argument file are stored
    e = 0
    log_dir = strftime("%d-%m-%y %H:%M:%S", gmtime())
    output_dir = os.path.join(output_dir, log_dir)
    if not os.path.exists(output_dir):
        print(f"{output_dir} directory created")
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "encoder"))
        os.makedirs(os.path.join(output_dir, "linear"))
        copyfile(config_file, os.path.join(output_dir, "params.yaml"))

    encoder.train()
    for linear in linear_list:
        linear.train()

    with open(config_file) as f:
        conf = yaml.safe_load(f)

    # inspector for the contrastive loss and for model performance in the validation set
    # if starts to overfit stop training and use encoder only for feat extraction
    inspector = ModelMonitoring(patience=conf["training"]["patience"])
    inspector_encoder = ModelMonitoring(patience=2000)
    stop_encoder = False

    optimizer_encoder, scheduler_encoder = optimizer[0], scheduler[0]
    optimizer_decoder, scheduler_decoder = optimizer[1], scheduler[1]
    n_classes = conf["dataset"]["classes"]

    if dataset == "CMU_MOSEI" or dataset == "CMU_MOSEI_RAW":
        label = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
    elif dataset == "IEMOCAP":
        if loader_train.dataset.n_labels == 6:
            label = ['excited', 'happy', 'neutral',
                     'frustrated', 'angry', 'sad']
        else:
            label = ['happy', 'neutral', 'angry', 'sad']

    elif dataset == "RAVDESS":
        label = ["neutral", "calm", "happy", "sad",
                 "angry", "fearful", "disgust", "surprised"]

    # to have correct metrics is importat to keep the sequence in that order
    # because is the order of the encoder, obviosuly can be change but then we need
    # to change the label order: standard is [af,ldf,vf,tf]
    sep_acc_mod = []
    if conf["dataset"]["audio"]:
        sep_acc_mod.append("Audio")
    if conf["dataset"]["graph"]:
        sep_acc_mod.append("Graph")
    if conf["dataset"]["video"]:
        sep_acc_mod.append("Video")
    if conf["dataset"]["text"]:
        sep_acc_mod.append("Text")

    if dataset == "CMU_MOSEI" or dataset == "CMU_MOSEI_RAW":
        # weight to balance the different emotions, for MOSEI BCE loss is used, otherwise contrastive loss
        w = torch.Tensor([1.05285, 5.215074, 5.3373003,
                         19.489855, 7.0448384, 23.688004]).to(device)
        classifier_loss = torch.nn.BCEWithLogitsLoss(pos_weight=w)
        sigm = torch.nn.Sigmoid()

    cmu_mosei_flag = dataset == "CMU_MOSEI" or dataset == "CMU_MOSEI_RAW"
    skip_contrastive = False
    # amp -> automatic mixed precision to reduce the computation time, if true is a bit faster
    # but more important allows to fit more samples in each batch
    use_amp = use_amp
    scaler = torch.cuda.amp.GradScaler()

    # if modality contrastive is false concatenate the modalities and contrast all the modalities
    # otherwise use the "clip-based" approach
    print(f"modality contrastive {modality_contrastive}")

    for e in range(epochs):

        samples = 0.
        batch_count = 0
        cumulative_loss = 0.
        cumulative_contr_loss = 0.
        cumulative_ce_loss = 0.

        cumulative_confidence_loss = 0.

        # (-1 is becasue on linear is for the combined features)
        sep_mod_pred = [torch.Tensor([]) for _ in range(len(linear_list)-1)]

        print(f"Epoch -  {e}  inspector {inspector_encoder.stopped}")
        tot_target = torch.Tensor([])
        tot_prediction = torch.Tensor([])

        for _, batch in enumerate(tqdm(loader_train)):

            targets = batch[0].to(device, non_blocking=True)
            features_s_1 = batch[1]
            features_s_2 = batch[2]
            # put the two sequences on the devices
            for feat_id, _ in enumerate(features_s_1):
                if len(features_s_1[feat_id]) != 0:
                    features_s_1[feat_id] = features_s_1[feat_id].to(
                        device, non_blocking=True).float()

            for feat_id, _ in enumerate(features_s_2):
                if len(features_s_2[feat_id]) != 0:
                    features_s_2[feat_id] = features_s_2[feat_id].to(
                        device, non_blocking=True).float()

            # if the encoder training is done stop it and use it only to extract features and finetute (didn't really change anything)
            if inspector_encoder.stopped:
                with torch.no_grad():
                    proj_feat_1, l_feat_1 = encoder(features_s_1)
                    proj_feat_2, l_feat_2 = encoder(features_s_2)
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    proj_feat_1, l_feat_1 = encoder(features_s_1)
                    proj_feat_2, l_feat_2 = encoder(features_s_2)

            # loop over each modality (-1 is becasue on linear is for the combined features)
            head_batch_pred = [None for _ in range(len(linear_list)-1)]
            if cmu_mosei_flag:
                # remove the neutral since is handcrafted only for the contrastive
                # and set the label to [0,1] as they do in literature instead of using the intensity
                # (might be interesting to try with intensity, there are basically no baselines)
                tgt = targets[:, 1:]
                tgt[tgt > 0] = 1
                tgt = torch.cat((tgt, tgt))
                tgt = tgt.float()
            else:
                tgt = targets
                tgt = torch.cat((tgt, tgt))

            # train the separate head predictions
            for feat_dim in range(len(l_feat_1)):
                optimizer_decoder[feat_dim+1].zero_grad()
                if inspector_encoder.stopped:
                    with torch.no_grad():
                        feat_separate = torch.cat(
                            (l_feat_1[feat_dim], l_feat_2[feat_dim]), 0)
                else:
                    feat_separate = torch.cat(
                        (l_feat_1[feat_dim].detach(), l_feat_2[feat_dim].detach()), 0)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    # +1 because 0 is the combined
                    logits = linear_list[feat_dim+1](feat_separate)
                    ce_loss = classifier_loss(logits, tgt)

                if use_amp:
                    scaler.scale(ce_loss).backward()
                    scaler.step(optimizer_decoder[feat_dim+1])
                else:
                    ce_loss.backward()
                    optimizer_decoder[feat_dim+1].step()

                if not cmu_mosei_flag:
                    _, logits = logits.max(1)
                else:
                    logits = sigm(logits).round()

                sep_mod_pred[feat_dim] = torch.cat(
                    (sep_mod_pred[feat_dim], logits.detach().cpu()), 0)
                head_batch_pred[feat_dim] = logits.detach()

            l_contr_feat = []
            predicted_features = torch.Tensor([]).to(device, non_blocking=True)

            # test where each modality inside the encoder has a prediction head that goes to another modality
            # doesn't work
            if encoder.prediction_head:
                # [b_size, n_modality, n_modality-1, feat_dim]
                predicted_modal_1 = proj_feat_1[-1].unsqueeze(-2)
                predicted_modal_2 = proj_feat_2[-1].unsqueeze(-2)
                predicted_features = torch.cat(
                    (predicted_modal_1, predicted_modal_2), -2)
                proj_feat_1 = proj_feat_1[:-1]
                proj_feat_2 = proj_feat_2[:-1]

            if modality_contrastive:
                # create the two projected features, normalized also as they do in literature
                # each modality in l_contr_feat has shape [b_size, n_views, dim_feat] and represent the feature
                # for each view and each modality
                for i in range(len(proj_feat_1)):
                    proj_1 = torch.nn.functional.normalize(
                        proj_feat_1[i], dim=-1).unsqueeze(1)
                    proj_2 = torch.nn.functional.normalize(
                        proj_feat_2[i], dim=-1).unsqueeze(1)
                    contr_feat = torch.cat((proj_1, proj_2), 1)
                    l_contr_feat.append(contr_feat)
            else:
                # creates two parallel tensors that are the two view of the modality concatenated
                contr_feat_1 = torch.Tensor([]).to(device=device)
                contr_feat_2 = torch.Tensor([]).to(device=device)
                for i in range(len(proj_feat_1)):
                    contr_feat_1 = torch.cat((contr_feat_1, proj_feat_1[i]), 1)
                    contr_feat_2 = torch.cat((contr_feat_2, proj_feat_2[i]), 1)

                contr_feat_1 = torch.nn.functional.normalize(
                    contr_feat_1, dim=-1)
                contr_feat_2 = torch.nn.functional.normalize(
                    contr_feat_2, dim=-1)

                contr_feat = torch.cat(
                    (contr_feat_1.unsqueeze(1), contr_feat_2.unsqueeze(1)), 1)
                l_contr_feat = [contr_feat, contr_feat]

            feat_1 = torch.Tensor([]).to(device, non_blocking=True)
            feat_2 = torch.Tensor([]).to(device, non_blocking=True)

            for i in range(len(l_feat_1)):
                feat_1 = torch.cat((feat_1, l_feat_1[i]), -1)
                feat_2 = torch.cat((feat_2, l_feat_2[i]), -1)

            if not skip_contrastive and not inspector_encoder.stopped:
                combined_feat = torch.cat(
                    (feat_1.detach(), feat_2.detach()), 0)
            else:
                combined_feat = torch.cat((feat_1, feat_2), 0)

            if cmu_mosei_flag:
                # remove, neutral emotion or the sentiment
                tgt = targets[:, 1:]
                tgt[tgt > 0] = 1
                tgt = tgt.float()
                tgt = torch.cat((tgt, tgt))
            else:
                tgt = torch.cat((targets, targets))

            with torch.cuda.amp.autocast(enabled=use_amp):

                logits, ll_features = linear_list[0](combined_feat)
                ce_loss = classifier_loss(logits, tgt)

            if not skip_contrastive and not inspector_encoder.stopped:
                # if we are in the unsupervised setting
                if conf["training"]["unsupervised"]:
                    n_losses = 0
                    contr_loss = 0

                    for idx_feat, feat in enumerate(l_contr_feat):
                        for idx_innner, feat_inner in enumerate(l_contr_feat):
                            # contrast all the modalities with the other modalities except if it is the same modality
                            if idx_feat != idx_innner:
                                with torch.cuda.amp.autocast(enabled=use_amp):
                                    contr_loss += encoder_loss(feat,
                                                               feat_inner)
                                n_losses += 1
                    # loss normalization
                    contr_loss /= n_losses
                else:
                    n_losses = 0
                    contr_loss = 0
                    # with the prediction head a bit different the correct prediction head has to be selected

                    if encoder.prediction_head:
                        for idx_feat in range(predicted_features.shape[1]):
                            pred_head_idx = 0
                            for idx_innner, feat_inner in enumerate(l_contr_feat):
                                tgt_contr = targets
                                if idx_feat != idx_innner:
                                    feat = predicted_features[:,
                                                              idx_feat, pred_head_idx]
                                    with torch.cuda.amp.autocast(enabled=use_amp):
                                        contr_loss += encoder_loss(
                                            feat, feat_inner, tgt_contr)
                                    pred_head_idx += 1
                                    n_losses += 1
                    else:
                        for idx_feat, feat in enumerate(l_contr_feat):
                            for idx_innner, feat_inner in enumerate(l_contr_feat):
                                tgt_contr = targets
                                # skip the last one if we are in a linear contrastive setting
                                if not(idx_feat == len(l_contr_feat)-1 ):
                                    if idx_feat != idx_innner:
                                        with torch.cuda.amp.autocast(enabled=use_amp):
                                            contr_loss += encoder_loss(
                                                feat, feat_inner, tgt_contr)
                                        n_losses += 1
                    contr_loss /= n_losses
            else:
                contr_loss = 0

            if not skip_contrastive and not inspector_encoder.stopped:
                optimizer_encoder.zero_grad()
                if use_amp:
                    scaler.scale(contr_loss).backward()
                    scaler.step(optimizer_encoder)
                else:
                    contr_loss.backward()
                    optimizer_encoder.step()

            optimizer_decoder[0].zero_grad()
            if use_amp:
                scaler.scale(ce_loss).backward()
                scaler.step(optimizer_decoder[0])
            else:
                ce_loss.backward()
                optimizer_decoder[0].step()

            if use_amp:
                scaler.update()

            loss = contr_loss + ce_loss

            batch_size = tgt.shape[0]
            samples += batch_size*2
            batch_count += 1
            cumulative_loss += loss.item()

            if not skip_contrastive and not inspector_encoder.stopped:
                # Note: the .item() is needed to extract scalars from tensors
                cumulative_contr_loss += contr_loss.item()
            else:
                cumulative_contr_loss += contr_loss
            # Note: the .item() is needed to extract scalars from tensors
            cumulative_ce_loss += ce_loss.item()

            # if we are using something that is not cmu-mosei pick the most likely otherwise
            # sigmoid and round
            if not cmu_mosei_flag:
                _, logits = logits.max(1)
            else:
                logits = sigm(logits).round()

            tot_prediction = torch.cat(
                (tot_prediction, logits.detach().cpu()), 0)
            tot_target = torch.cat((tot_target, tgt.cpu()), 0)

        if conf["training"]["unsupervised"]:
            # treshold and alpha are for the pseudolabelling that didn't work
            alpha = 2
            treshold = max(85, 99-e*alpha)
            encoder_train(encoder, loader_valid, optimizer_encoder, encoder_loss, linear_list, classifier_loss, optimizer_decoder, device=device, scaler=scaler,
                          use_amp=use_amp, split="Valid", wandb=wandb, dataset=dataset, treshold=treshold, modality_contrastive=modality_contrastive)
            encoder_train(encoder, loader_test, optimizer_encoder, encoder_loss, linear_list, classifier_loss, optimizer_decoder, device=device, scaler=scaler,
                          use_amp=use_amp, split="Test", wandb=wandb, dataset=dataset, treshold=treshold, modality_contrastive=modality_contrastive)

        if not skip_contrastive:
            scheduler_encoder.step()

        for sch_dec in scheduler_decoder:
            sch_dec.step()

        final_loss = cumulative_loss/batch_count
        final_contr_loss = cumulative_contr_loss/batch_count
        final_ce_loss = cumulative_ce_loss/batch_count
        if confidence:
            final_conf_loss = cumulative_confidence_loss/batch_count

        if cmu_mosei_flag:
            # CMU_MOSEI -> compute F1 weighted score and accuracy for each modality and for each labeal
            tot_target = tot_target.numpy()
            tot_prediction = tot_prediction.numpy()
            # compute the accuracy and f1 score for each label
            label_f1 = []
            label_accuracy = []
            for idx in range(tot_target.shape[1]):
                f1 = f1_score(
                    tot_target[:, idx], tot_prediction[:, idx], average='weighted')
                #f1 = f1_score(tot_target[:,idx], tot_prediction[:,idx])
                label_f1.append(f1)
                acc = 100 * \
                    accuracy_score(tot_target[:, idx], tot_prediction[:, idx])
                label_accuracy.append(acc)

            f1 = mean(label_f1)
            accuracy = mean(label_accuracy)
            # compute accuracy and score for the heads
            accuracy_sep_mod = []
            f1_sep_mod = []

            for mod in sep_mod_pred:
                mod = mod.numpy()
                f1_l_sep_mod = []
                acc_l_sep_mod = []
                for idx in range(tot_target.shape[1]):
                    f1 = f1_score(tot_target[:, idx],
                                  mod[:, idx], average='weighted')
                    acc = 100 * accuracy_score(tot_target[:, idx], mod[:, idx])
                    acc_l_sep_mod.append(acc)
                    f1_l_sep_mod.append(f1)
                f1_sep_mod.append(mean(f1_l_sep_mod))
                accuracy_sep_mod.append(mean(acc_l_sep_mod))
        else:
            # computation of f1 and acc for a dataset different than CMU-MOSEI
            tot_target = tot_target.numpy()
            tot_prediction = tot_prediction.numpy()
            f1 = f1_score(tot_target, tot_prediction, average='weighted')
            matrix = confusion_matrix(tot_target, tot_prediction)
            label_accuracy = matrix.diagonal()/matrix.sum(axis=1)
            accuracy = 100 * accuracy_score(tot_target, tot_prediction)

            # since is categorical here we log also the confusion matrix
            if wandb is not None:
                wandb.log({"train_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                                         y_true=tot_target, preds=tot_prediction,
                                                                         class_names=label)})

            accuracy_sep_mod = []
            for mod in sep_mod_pred:
                accuracy_mod = 100 * accuracy_score(tot_target, mod.numpy())
                accuracy_sep_mod.append(accuracy_mod)

        # test performance over the test set, if e%1 is changed avoid to aggressively evaluate after each epoch
        if test and e % 1 == 0:
            # test the model on the validation set
            valid_loss, valid_contr_loss, valid_ce_loss, valid_accuracy, valid_label_accuracy, valid_label_f1, valid_accuracy_separate, valid_f1 = evaluate_model(
                encoder, linear_list, loader_valid, encoder_loss, classifier_loss, dataset=dataset, device=device, unsupervised=conf["training"]["unsupervised"], wandb=wandb, n_classes=n_classes, use_amp=use_amp, split="Valid", label=label)
            inspector(valid_accuracy)
            inspector_encoder(-1*valid_contr_loss)
            # log performance on validation set
            log_evaluation_performance(valid_loss, valid_contr_loss, valid_ce_loss, valid_accuracy, valid_label_accuracy, valid_label_f1,
                                       valid_accuracy_separate, valid_f1, set="Valid", label=label, wandb=wandb, sep_acc_mod=sep_acc_mod, dataset=dataset)
            # if the performance on the validation is improved run on test and save the checkpoint
            if inspector.counter == 0:
                test_loss, test_contr_loss, test_ce_loss, test_accuracy, test_label_accuracy, test_label_f1, test_accuracy_separate, test_f1 = evaluate_model(
                    encoder, linear_list, loader_test, encoder_loss, classifier_loss, dataset=dataset, device=device, unsupervised=conf["training"]["unsupervised"],  wandb=wandb, n_classes=n_classes, use_amp=use_amp, split="Test", label=label)
                log_evaluation_performance(test_loss, test_contr_loss, test_ce_loss, test_accuracy, test_label_accuracy, test_label_f1,
                                           test_accuracy_separate, test_f1, set="Test", label=label, wandb=wandb, sep_acc_mod=sep_acc_mod, dataset=dataset)

                filename = os.path.join(
                    output_dir, "encoder", "encoder_epoch_top.pth")
                torch.save({
                    'epoch': e,
                    'model_state_dict': encoder.state_dict(),
                    'loss': final_loss,
                }, filename)
                filename = os.path.join(
                    output_dir, "linear", "linear_epoch_top.pth")

                torch.save({
                    'epoch': e,
                    'model_state_dict': linear[0].state_dict(),
                    'loss': final_loss,
                }, filename)

            if inspector_encoder.counter == 0 and not stop_encoder:
                print("COPY BEST ENCODER")
                best_encoder = copy.deepcopy(encoder)
                best_encoder = best_encoder.to('cpu')

        if not inspector_encoder.stopped and not skip_contrastive:
            print(
                f"Inspector encoder {inspector_encoder.counter}/{inspector_encoder.patience} contro_loss {inspector_encoder.best_score}")

        # print and log the training loop
        print(
            f"BEST SCORE {inspector.best_score} Test accuracy {test_accuracy} count {inspector.counter}/{inspector.patience}")
        print('\t Training loss {:.5f}, Train_contr_loss {:.5f}, Train_ce_loss {:.5f}, Training accuracy {:.2f}'.format(
            final_loss, final_contr_loss, final_ce_loss, accuracy))
        if wandb is not None:
            wandb.log({"Accuracy": accuracy, "Contrastive Loss": final_contr_loss,
                       "Cross Entropy Loss": final_ce_loss,  "F1": f1, "Total Loss": final_loss})

            for idx, acc in enumerate(accuracy_sep_mod):
                wandb.log({"Accuracy_"+sep_acc_mod[idx]: acc})
                if cmu_mosei_flag:
                    wandb.log({"F1_"+sep_acc_mod[idx]: f1_sep_mod[idx]})

            for i, lab_acc in enumerate(label_accuracy):
                wandb.log({"Train_label_percentage_"+str(label[i]): lab_acc})
                if cmu_mosei_flag:
                    wandb.log({"Train_label_f1_"+str(label[i]): label_f1[i]})

        if inspector.stopped:
            print(
                f"BEST SCORE {inspector.best_score}  Test accuracy {test_accuracy} ")

            break

        if inspector_encoder.stopped and not stop_encoder and not skip_contrastive:
            # load the encoder weights inside the module and set to evaluate
            encoder = best_encoder
            encoder = encoder.to(device)
            encoder.eval()

            stop_encoder = True
            print(f"ENCODER TRAINED----------- epoch {e}")
