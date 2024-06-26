import argparse
import os
import wandb

from dotenv import load_dotenv

load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--root", type=str, default=os.environ.get("MAIN"))
    # wandb
    parser.add_argument("--wandb_project", type=str, default="final")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_disable", action="store_true")
    # data
    parser.add_argument("--dataset", type=str, default="bars") #bars
    # parser.add_argument("--data_dir", type=str,
    #                     default="features/i3d_embeddings/")
    parser.add_argument("--data_dir", type=str, default="rgb-images")
    # use images instead of image sequences
    parser.add_argument("--flat", action="store_true")
    # use bounding box data (available for UCF101-24)
    parser.add_argument("--bboxes", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    # replace the frames with random noise
    parser.add_argument("--random", action="store_true")
    # replace the frames with indices reshaped as random noise
    parser.add_argument("--indices", action="store_true")
    # normalize the indices by dividing by this value
    parser.add_argument("--indices_normalizer", type=float, default=1.0)
    # use subsampling as is described in "Am I Done" https://arxiv.org/abs/1705.01781
    parser.add_argument("--subsample", action="store_true")
    # max length of a sequence, anything after this will be cutoff
    parser.add_argument('--max_length', type=int, default=1000000)
    # subsample sequences to avoid sequences becoming too long
    parser.add_argument("--subsample_fps", type=int, default=1)
    # set either no Remaining Surgery Duration (RSD), RSD in minutes, or RSD in seconds
    parser.add_argument(
        "--rsd_type", type=str, default="none", choices=["none", "minutes", "seconds"]
    )
    # RSD normalizer value, this is the s_norm value described in RSDNet, https://arxiv.org/pdf/1802.03243.pdf
    parser.add_argument("--rsd_normalizer", type=float, default=1.0)
    # fps of the data, used to calculate RSD
    parser.add_argument("--fps", type=float, default=1.0)
    # disable resizing
    parser.add_argument("--no_resize", action="store_true")
    parser.add_argument("--train_split", type=str, default="train.txt")
    parser.add_argument("--test_split", type=str, default="test.txt")
    # training
    parser.add_argument("--batch_size", type=int, default=1) #40
    parser.add_argument("--iterations", type=int, default=10000)
    # network
    parser.add_argument(
        "--network",
        type=str,
        default="progressnet", # progressnet
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='resnet18',
        choices=["vgg16", 'vgg11', "resnet18", "resnet152", "swintransformer", "resnext50"],
    )
    parser.add_argument("--load_backbone", type=str, default='resnet18.pth')
    # network parameters
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--dropout_chance", type=float, default=0.0)
    parser.add_argument("--pooling_layers", nargs="+",
                        type=int, default=[1, 2, 3])
    parser.add_argument("--roi_size", type=int, default=3)
    parser.add_argument('--finetune', action='store_true')
    # network loading
    parser.add_argument("--load_experiment", type=str, default=None)  # pn_bars
    parser.add_argument("--load_iteration", type=int, default=None)  # 1000
    # optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument(
        "--loss", type=str, default="l2", choices=["l2", "l1", "smooth_l1"]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    # scheduler
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay_every", type=int, default=503 * 30)
    # logging
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--test_every", type=int, default=1000)

    parser.add_argument("--print_only", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--embed_batch_size", type=int, default=10)
    parser.add_argument("--embed_dir", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def wandb_init(args):
    #     no_wandb = (
    #         args.wandb_disable or args.print_only or args.eval or args.embed or (
    #             args.save_dir is not None)
    #     )
    #     if no_wandb:
    #         return
    #
    #     # TODO: Config = args (possibly on reruns)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=args,
        settings=wandb.Settings(start_method="fork")
    )
