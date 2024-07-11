import sys
import os
from argparse import ArgumentParser
from torch.distributed.run import run as td_run, parse_args as tdr_parse_args

sys.path.append("./mae")
from mae.main_pretrain import main as mae_main, get_args_parser as mae_argparser
sys.path.append("./segmenter")
from segmenter.segm.train import main as seg_main
from segmenter.segm.scripts.prepare_ade20k import main as ade_main



def get_args_parser():
    parser = ArgumentParser('MAE pretraining and Segmenter evaluation')
    parser.add_argument('--in1k-dir', required=True, type=str,
                        help="Directory containing the Imagenet-1k dataset. This should have 'train', 'val' and 'masks' subfolders")
    parser.add_argument('--output-dir', default=".", type=str,
                        help="Directory to output model checkpoints and logs to. Default: current directory")
    parser.add_argument('--mask-method', default='patches', type=str,
                        help='Masking method to use (options: patches, segments, four_channels, preencoder).')
    parser.add_argument('--coverage-ratio', default=15.0, type=float,
                        help='Coverage percentage to strive for when selecting segmentation masks for pretraining')
    parser.add_argument('--num-gpus', default=1, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--pretrain-epochs', default=0, type=int,
                        help='Number of epochs to test pretraining for')
    parser.add_argument('--eval-freq', default=5, type=int,
                        help='Number of epochs to pretrain before each ADE20k evaluation run')

    return parser

def get_mae_args(img1k_dir: str, output_dir: str, mask_method: str, coverage_ratio: float, gpu: int, epochs: int, resume: str) -> list[str]:
    arglist = [
        "--batch_size", "32",
        "--epochs", str(epochs),
        "--mask_method", mask_method,
        "--coverage_ratio", str(coverage_ratio),
        "--blr", str(1.5e-4),
        "--output_dir", f"{output_dir}/pretrain_output/{mask_method}",
        "--log_dir", "",
        "--data_path", img1k_dir,
        "--resume", resume,
    ]

    if gpu >= 0:
        arglist += ["--device", f"cuda:{gpu}"]

    return arglist

def get_seg_args(mask_method: str, epochs: int, gpu: int) -> dict:
    kwargs = []
    kwargs += ["--log-dir", f"{args.output_dir}/eval_output/{mask_method}"]
    kwargs += ["--dataset", "ade20k"]
    kwargs += ["--crop-size", "224"]
    kwargs += ["--backbone", "vit_large_mae"]
    kwargs += ["--decoder", "mask_transformer"]
    kwargs += ["--mae"]
    kwargs += ["--mae_chp", f"{args.output_dir}/pretrain_output/{args.mask_method}/checkpoint-{epochs}.pth"]
    if mask_method == "four_channels":
        kwargs += ["--channels", "4"]
    if gpu >= 0:
        os.environ["SLURM_LOCALID"] = str(gpu)
    return kwargs

def main(args):
    if not os.path.isdir("segmenter/data/ade20k"):
        ade_main(["segmenter/data"])

    if args.mask_method not in ["patches", "segments", "four_channels", "preencoder"]:
        print(f"Unknown mask method \"{args.mask_method}\". Expected one of \"patches\", \"segments\", \"four_channels\", \"preencoder\"")
        exit(1)

    if args.num_gpus > 1:
        args.gpu = -1

    runs = args.pretrain_epochs//args.eval_freq
    for i in range(runs):
        resume = ""
        if i:
            resume = f"{args.output_dir}/pretrain_output/{args.mask_method}/checkpoint-{epochs}.pth"
        epochs = args.eval_freq * (i+1)
        print(f"Training run {i+1}/{runs} for {epochs} epochs")
        mae_args = get_mae_args(args.in1k_dir, args.output_dir, args.mask_method, args.coverage_ratio, args.gpu, epochs, resume)
        if args.num_gpus > 1:
            tdr_args = ["--standalone", "--nnodes", "1", "--nproc-per-node", str(args.num_gpus)]
            tdr_args += ["mae/main_pretrain.py"]
            tdr_args += mae_args
            tdr_args = tdr_parse_args(tdr_args)
            os.environ["OMP_NUM_THREADS"] = "1"
            td_run(tdr_args)
        else:
            mae_args = mae_argparser().parse_args(mae_args)
            mae_main(mae_args)

        os.rename(
            f"{args.output_dir}/pretrain_output/{args.mask_method}/checkpoint-{epochs-1}.pth",
            f"{args.output_dir}/pretrain_output/{args.mask_method}/checkpoint-{epochs}.pth")

        print(f"Evaluating run {i+1}/{runs}")
        seg_args = get_seg_args(args.mask_method, epochs, args.gpu)
        if args.num_gpus > 1:
            tdr_args = ["--standalone", "--nnodes", "1", "--nproc-per-node", str(args.num_gpus)]
            tdr_args += ["segmenter/segm/train.py"]
            tdr_args += seg_args
            tdr_args = tdr_parse_args(tdr_args)
            os.environ["OMP_NUM_THREADS"] = "1"
            td_run(tdr_args)
        else:
            seg_main.main(seg_args, standalone_mode=False)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
