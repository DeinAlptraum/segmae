import sys
import os
from pathlib import Path
from shutil import copyfile
from subprocess import run as sub_run
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
    parser.add_argument('--stack', action='store_true',
                        help='Continue training on top of the MAE paper\'s pretrained checkpoint')
    parser.add_argument('--eval', default=-1, type=int,
                        help='Evaluate the model by fine-tuning on ADE20k')

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
    kwargs += ["--log-dir", f"{args.output_dir}/eval_output/{mask_method}_{epochs}"]
    kwargs += ["--dataset", "ade20k"]
    kwargs += ["--crop-size", "224"]
    kwargs += ["--backbone", "vit_large_mae"]
    kwargs += ["--decoder", "mask_transformer"]
    kwargs += ["--mae"]
    if epochs == -2:
        kwargs += ["--no-resume"]
    if epochs != -3:
        kwargs += ["--mae_chp", f"{args.output_dir}/pretrain_output/{args.mask_method}/checkpoint-{epochs}.pth"]
    else:
        kwargs += ["--mae_chp", f"./mae_pretrain_vit_large.pth"]
    if mask_method == "four_channels":
        kwargs += ["--channels", "4"]
    if gpu >= 0:
        os.environ["LOCAL_RANK"] = str(gpu)
    return kwargs

CHECKPOINT_NAME = "mae_pretrain_vit_large.pth"

def main(args):
    # Download ADE20k if not already there
    if not os.path.isdir("segmenter/data/ade20k"):
        ade_main(["segmenter/data"])

    # If training on top of the pretrained checkpoint, download it as well
    if args.stack and not os.path.isfile(CHECKPOINT_NAME):
        print("Downloading pretrained checkpoint...")
        sub_run(["wget", f"https://dl.fbaipublicfiles.com/mae/pretrain/{CHECKPOINT_NAME}"])

    if args.mask_method not in ["patches", "segments", "four_channels", "preencoder"]:
        print(f"Unknown mask method \"{args.mask_method}\". Expected one of \"patches\", \"segments\", \"four_channels\", \"preencoder\"")
        exit(1)

    if args.num_gpus > 1:
        args.gpu = -1

    if args.stack:
        epochs = 0
        subfolder = Path(f"{args.output_dir}/pretrain_output/{args.mask_method}")
        Path(subfolder).mkdir(parents=True, exist_ok=True)
        copyfile(CHECKPOINT_NAME, subfolder/"checkpoint-0.pth")


    if args.eval == -1:
        runs = args.pretrain_epochs//args.eval_freq
        for i in range(runs):
            resume = ""
            if i or args.stack:
                resume = f"{args.output_dir}/pretrain_output/{args.mask_method}/checkpoint-{epochs}.pth"
            epochs = args.eval_freq * (i+1)
            print(f"Training run {i+1}/{runs} for {epochs} epochs")
            mae_args = get_mae_args(args.in1k_dir, args.output_dir, args.mask_method, args.coverage_ratio, args.gpu, epochs, resume)
            if args.stack and i == 0:
                mae_args += ["--stack"]

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
    else:
        epochs = args.eval
        chp = f"{args.output_dir}/pretrain_output/{args.mask_method}/checkpoint-{epochs}.pth"
        if not Path(chp).is_file() and epochs >= -1:
            print("The checkpoint to be evaluated does not exist: ", chp)
            exit(1)
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
