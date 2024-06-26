import numpy as np
import os
import random
import torch
import wandb

from arguments import parse_args, wandb_init
from datasets import ImageDataset, UCFDataset, Subsample, Subsection, Truncate
from dotenv import load_dotenv
from experiment import Experiment, get_device
from networks import ProgressNet
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from train_functions import train_flat_features, train_flat_frames, train_progress, embed_frames

load_dotenv()

def set_seeds(seed: int) -> None:
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# def collate(batch):
#    batch = filter(lambda x: x is not None, batch)
#    print()
#    return torch.utils.data.dataloader.default_collate(batch)

# def collate_fn_padd(batch):
#     '''
#     Padds batch of variable length
# 
#     note: it converts things ToTensor manually here since the ToTensor transform
#     assume it takes in images rather than arbitrary tensors.
#     '''
#     device = get_device()
#     ## get sequence lengths
#     lengths = torch.tensor([int(t[1].shape[0]) for t in batch ]).to(device)
#     ## padd
#     #batch = [ t[2].to(device) for t in batch ]
#     batch = torch.nn.utils.rnn.pad_sequence([batch], padding_value = 0)
#     ## compute mask
#     # mask = (batch != 0).to(device)
#     return batch#, lengths, mask

def main():
    args = parse_args()
    set_seeds(args.seed)

    wandb.login()
    torch.cuda.empty_cache()
    # root can be set manually, but can also be obtained automatically so wandb sweeps work properly
    if args.root is not None:
        root = args.root
    else:
        root = os.environ.get('MAIN')
    data_root = os.path.join(root, args.dataset)

    # setup experiment
    experiment_path = None
    if args.experiment_name and args.experiment_name.lower() != "none":
        experiment_path = os.path.join(
            root, "experiments/", args.experiment_name)

    # setup sample transform
    if args.subsample:
        subsample = transforms.Compose(
            [Subsection(), Subsample(), Truncate(args.max_length)])
    else:
        subsample = None

    # create datasets
    # TODO: Combine datasets and clean up code
    # datasets can be roughly grouped into the following categories:
    # UCFDataset: Dataset specifically made for the bounding boxes available in UCF101-24
    # ImageDataset: Dataset using image (sequences).
    # FeatureDataset: Dataset using image (sequence) embeddings, i.e. embedding vectors.
    if "images" in args.data_dir:
        transform = [transforms.ToTensor()]
        if not args.no_resize:
            transform.append(transforms.Resize((224, 224), antialias=True))
        transform = transforms.Compose(transform)

        if "ucf24" in args.dataset:
            trainset = UCFDataset(
                data_root,
                args.data_dir,
                args.train_split,
                args.bboxes,
                args.flat,
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.rsd_type,
                args.fps,
                transform=transform,
                sample_transform=subsample,
            )
            testset = UCFDataset(
                data_root,
                args.data_dir,
                args.test_split,
                args.bboxes,
                args.flat,
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.rsd_type,
                args.fps,
                transform=transform,
            )
        else:
            trainset = ImageDataset(
                data_root,
                args.data_dir,
                args.train_split,
                args.flat,
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.shuffle,
                transform=transform,
                sample_transform=subsample,
            )
            testset = ImageDataset(
                data_root,
                args.data_dir,
                args.test_split,
                args.flat,
                args.subsample_fps,
                args.random,
                args.indices,
                args.indices_normalizer,
                args.shuffle,
                transform=transform,
            )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True# , collate_fn=collate_fn_padd
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False# , collate_fn=collate_fn_padd
    )

    # load backbone and create the network
    # TODO: Reorganise networks
    if args.load_backbone:
        backbone_path = os.path.join(
            data_root, "train_data", args.load_backbone)
    else:
        backbone_path = None

    print(backbone_path)
    if args.network == "progressnet":
        network = ProgressNet(
            args.pooling_layers,
            args.roi_size,
            args.dropout_chance,
            args.embed_dim,
            args.finetune,
            args.backbone,
            backbone_path,
        )
    else:
        raise Exception(f"Network {args.network} does not exist")

    # load network file if available
    if args.load_experiment and args.load_iteration:
        network_path = os.path.join(
            root,
            "experiments",
            args.load_experiment,
            f"model_{args.load_iteration}.pth",
        )
        network.load_state_dict(torch.load(network_path))

    # create optimizer, scheduler, and loss
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            network.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            network.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    else:
        raise Exception(f"Optimizer {args.optimizer} does not exist")

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_every, args.lr_decay)

    if args.loss == "l2":
        criterion = nn.MSELoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        raise Exception(f"Loss {args.loss} does not exist")

    # Get the training function used for the combination of network and dataset.
    # This is a similar concept to pytorch lightning. A lot of the training/testing/logging logic remains the same
    # and only the train/test function is swapped out.
    # This part also creates a default dictionary which is used for wandb logging.
    # TODO: Redo
    result = {"l1_loss": 0.0, "l2_loss": 0.0, "count": 0}
    if args.embed:
        train_fn = None
    elif "images" not in args.data_dir and args.flat:
        train_fn = train_flat_features
    elif "images" in args.data_dir and args.flat:
        train_fn = train_flat_frames
    else:
        train_fn = train_progress

    wandb_init(args)
    # create and start the experiment
    experiment = Experiment(
        network,
        criterion,
        optimizer,
        scheduler,
        trainloader,
        testloader,
        train_fn,
        args.max_length,
        experiment_path,
        result,
    )
    experiment.print()
    if args.eval:
        experiment.eval()
    elif args.embed:
        if args.flat:
            raise Exception("Can't embed flat dataset")
        network.eval()
        with torch.no_grad():
            save_dir = os.path.join(data_root, args.embed_dir)
            os.makedirs(save_dir, exist_ok=True)
            for batch in tqdm(trainloader):
                video_name, embeddings = embed_frames(
                    network, batch, experiment.device, args.embed_batch_size
                )
                txt = []
                for embedding in embeddings:
                    txt.append(" ".join(map(str, embedding)))
                save_path = os.path.join(save_dir, f"{video_name}.txt")
                recursive_dir = '/'.join(save_path.split('/')[:-1])
                os.makedirs(recursive_dir, exist_ok=True)
                with open(save_path, "w+") as f:
                    f.write("\n".join(txt))
    elif args.save_dir is not None:
        if not os.path.isdir(f'./data/{args.save_dir}'):
            os.makedirs(f'./data/{args.save_dir}', exist_ok=True)
        experiment.save(args.save_dir)

    elif not args.print_only:
        experiment.run(args.iterations, args.log_every, args.test_every)

    wandb.finish()


if __name__ == "__main__":
    main()
