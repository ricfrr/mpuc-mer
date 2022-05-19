import torch
from argparse import ArgumentParser
import yaml

from models.Encoder import Encoder
from models.LinearClassifier import LinearClassifier

from utils.SupConSeparate import SupConLoss
from utils.utils import split_dataset

from data.RAVDESS import RAVDESS
from data.CMU_MOSEI import CMU_MOSEI
import os
from torch.utils.data import WeightedRandomSampler
from train import train


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.n_frames is not None:
        config["dataset"]["n_frames"] = args.n_frames

    if args.checkpoint is not None:
        with open(os.path.join(args.checkpoint, "params.yaml")) as f:
            config_checkpoint = yaml.safe_load(f)
        config["model_params"]["out_feat"] = config_checkpoint["model_params"]["out_feat"]
        config["dataset"]["video"] = config_checkpoint["dataset"]["video"]
        config["dataset"]["audio"] = config_checkpoint["dataset"]["audio"]
        config["dataset"]["text"] = config_checkpoint["dataset"]["text"]
        config["dataset"]["graph"] = config_checkpoint["dataset"]["graph"]

    if bool(args.wandb):
        import wandb
        wandb.init(project="fer",  config={
            "learning_rate_encoder": config["training"]["lr_encoder"],
            "learning_rate_classif": config["training"]["lr_linear"],
            "frame_l": config["dataset"]["n_frames"],
            "dataset": args.dataset
        })
    else:
        wandb = None

    sampler = None

    print(f"------ Initializing dataset...")
    if args.dataset == "CMU_MOSEI":
        dataset_train = CMU_MOSEI(
            config["dataset"]["path_train"], n_frames=config["dataset"]["n_frames"], augment=args.augment)
        dataset_test = CMU_MOSEI(
            config["dataset"]["path_test"], test=True, n_frames=config["dataset"]["n_frames"])
        dataset_valid = CMU_MOSEI(
            config["dataset"]["path_valid"], test=True, n_frames=config["dataset"]["n_frames"])
        _, samples_weight = dataset_train.get_class_sample_count()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    elif args.dataset == "RAVDESS":
        sample_test, sample_train, sample_valid = split_dataset(
            path=config["dataset"]["path"], perc=config["dataset"]["split_percentage"], path_audio=config["dataset"]["path_audio"], actor_split=config["dataset"]["actor_split"])
        dataset_train = RAVDESS(config["dataset"]["path"], samples=sample_train,
                                min_frames=config["dataset"]["n_frames"], n_mels=config["dataset"]["n_mels"])
        dataset_test = RAVDESS(config["dataset"]["path"], samples=sample_test,
                               min_frames=config["dataset"]["n_frames"], n_mels=config["dataset"]["n_mels"], test=True)
        dataset_valid = RAVDESS(config["dataset"]["path"], samples=sample_valid,
                                min_frames=config["dataset"]["n_frames"], n_mels=config["dataset"]["n_mels"], test=True)
        _, samples_weight = dataset_train.get_class_sample_count()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    if sampler is None:
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"], drop_last=False)
    else:
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"], drop_last=False, sampler=sampler)

    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"], drop_last=False)
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"], drop_last=False)
    print(f"------ Dataset Initialized")

    print(f"------ Initializing networks")
    cmu_mosei_flag = args.dataset == "CMU_MOSEI" or args.dataset == "CMU_MOSEI_RAW"
    encoder = Encoder(config_file=args.config, cmu_mosei=cmu_mosei_flag, device=args.device,
                      n_frames=config["dataset"]["n_frames"], prediction_head=args.pred_head)
    modalities = [config["dataset"]["video"], config["dataset"]
                  ["audio"], config["dataset"]["text"], config["dataset"]["graph"]]
    num_multimodal_input = sum(modalities)

    linear_layers = []

    linear = LinearClassifier(num_feat=config["model_params"]["out_feat"], n_modality=num_multimodal_input,
                              n_classes=config["dataset"]["classes"], attention=args.attention)

    linear = linear.to(args.device)
    linear_layers.append(linear)

    # create liner layer for each separate modality
    for _ in range(num_multimodal_input):
        ll = torch.nn.Sequential(torch.nn.Linear(config["model_params"]["out_feat"], config["model_params"]["out_feat"]), torch.nn.ReLU(
        ), torch.nn.Linear(config["model_params"]["out_feat"], config["dataset"]["classes"]))
        ll = ll.to(args.device)
        linear_layers.append(ll)

    encoder = encoder.to(args.device)
    print(f"------ Networks Initialized")

    print(f"------ Creating the optimizers")
    # create the SGD optimizer + Linear scheduler
    optimizer_encoder = torch.optim.SGD(encoder.parameters(
    ), config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(
        optimizer_encoder, step_size=config["training"]["scheduler_step"], gamma=config["training"]["scheduler_gamma"])

    optimizer_decoder_list = []
    scheduler_decoder_list = []

    # we perform the training also for each modality separately, therefore we instatiace n different optimizer and scheduler
    for linear in linear_layers:
        optimizer_decoder = torch.optim.SGD(linear.parameters(
        ), config["training"]["lr_linear"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
        scheduler_decoder = torch.optim.lr_scheduler.StepLR(
            optimizer_decoder, step_size=config["training"]["scheduler_step"], gamma=config["training"]["scheduler_gamma"])
        optimizer_decoder_list.append(optimizer_decoder)
        scheduler_decoder_list.append(scheduler_decoder)

    # loss
    print(f"------ Creating the losses")
    encoder_loss = SupConLoss(
        temperature=args.temperature, mask_temperature=args.mask_temperature)
    classifier_loss = torch.nn.CrossEntropyLoss()

    if args.checkpoint is not None:
        print("loading checkpoint")
        encoder.load_state_dict(torch.load(os.path.join(
            args.checkpoint, "encoder", "encoder_epoch_top.pth"), map_location=args.device)["model_state_dict"])
        linear[0].load_state_dict(torch.load(os.path.join(
            args.checkpoint, "linear", "linear_epoch_top.pth"), map_location=args.device)["model_state_dict"])

    # train
    print(f"------ Training Started")
    print(args.modality_contrastive)
    modality_contrastive = args.modality_contrastive == "True"
    train(encoder, linear_layers, loader_train, [optimizer_encoder, optimizer_decoder_list], [scheduler_encoder, scheduler_decoder_list], encoder_loss, classifier_loss, wandb, epochs=config["training"]["epochs"], device=args.device, test=True, use_amp=args.amp,
          loader_test=loader_test, loader_valid=loader_valid, output_dir=args.output,  dataset=args.dataset,  config_file=args.config,  modality_contrastive=modality_contrastive)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:2', type=str, help='device')
    parser.add_argument('--dataset', default='CMU_MOSEI_RAW',
                        type=str, help='daset name')
    parser.add_argument('--output', default=None, required=True,
                        help='folder where to store the ckp')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='temperature of supervised contrastive loss')
    parser.add_argument('--mask_temperature', default=0.01, type=float,
                        help='mask temperature in supervised contrastive loss')
    parser.add_argument("--n_frames", default=None,
                        type=int, help="len of the sequence")
    parser.add_argument("--attention", default=False, type=bool,
                        help="if multihead attention on combined features")
    parser.add_argument("--modality_contrastive", default="True",
                        type=str, help="if true contrast modalities in a CLIP fashion")
    parser.add_argument("--pred_head", default=False, type=bool,
                        help="if true enable prediction head inside the contrastive")
    parser.add_argument('--config', default=None, required=True,
                        type=str, help='path to config file')
    parser.add_argument('--amp', default=False, type=bool,
                        help='automatic mixed precision')
    parser.add_argument('--augment', default=True, type=bool,
                        help='if true applies image data augmentation')
    parser.add_argument('--checkpoint', default=None,
                        type=str, help='path to the checkpoint folder')
    parser.add_argument('--wandb', default=False, type=bool,
                        help='if false wandb doesnt log| defeault True')

    args = parser.parse_args()
    main(args)
