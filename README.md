This is the super-repo collecting
- a modification of https://github.com/facebookresearch/mae, the PyTorch implementation of the MAE paper. This contains the main pre-training method.
- a modification of https://github.com/rstrudel/segmenter, implementing a ViT-based semantic segmentation approach. This contains the ADE20k training task we use for evaluation.
- a benchmark script for combining these two into an integrated training & evaluation pipeline

The MAE and Segmenter repos are both submodules to this one, which only contains the dependency information and the benchmarking script gluing the two submodules together into a single pipeline. This will alternate between running IN1k pretraining for a few epochs, then evaluating that model through full training on ADE20k semantic segmentation, then training the checkpoint for more epochs.



Modifications include several bug fixes and modernizations for newer module versions, without which the code was not executable.

Further, I've implemented three different masking methods based on Segment-Anything masks:
- segments: multiplies inputs with the mask
- four_channels: multiplies inputs with the mask, and adds the mask itself as a fourth input channel. Expand the input dimension of the ViT model with a fourth channel, which is set to be always 1 after pre-training
- preencoder: add mask as fourth input channel like in four_channels, but compress this down to a 3-channel input volume of otherwise same size. This is done through a "pre-encoder" using a convolution with kernel size equaling the token patch size. The pre-encoder is thrown away after pretraining

### Environment setup
I would recommend Pipenv for the environment, as I tried with Conda but Conda's PyTorch-GPU package was broken at that time. I've included the Pipfile and Pipfile.lock to set this up, but also dumped the packages I used into the requirements.txt, though I cannot guarantee that these work. Note that it contains the index for torch on Cuda 11.8, which you might have to change, depending on the local cuda driver version.

I ran everything on Python 3.10.

### Running a benchmark
An example of how to run the pretraining on a single GPU can be seen in `run_benchmark.sh`.
The most important parts to adapt here are
- the `--in1k-dir` to the Imagenet-1K dataset. This should be the directory containing the `train`, `val` and `masks` folders, which in turn consist of the training and validation sets, and the Segment-Anything masks for the training set respectively. You can find the dataset on Panther in `/mnt/data0/jannick/img1k`
- the `--output-dir` path is where outputs are placed. By default, this is the current directory. Folder `pretrain_output` and `eval_output` will be created in that folder, which will contain the model checkpoints and logs of the MAE/Segmenter parts of the pipeline respectively
- the `--mask_method` you want to test, as one of `segments`, `four_channels`, `preencoder`, `patches`, where the latter is the original paper's implementation. I would recommend `four_channels` for the first test, or `patches` since we need to compare with the original model's performance _during_ pretraining anyway
- the `--coverage-ratio`, in percent, that we aim for when selecting a mask for an example. The mask with the coverage ratio closest to the given value will be picked.
- the `--pretrain-epochs` and `--eval-freq`. The former is the number of epochs that we pretrain for in total, while the latter specifies the interval at which to evaluate the pretraining performance by finetuning on the ADE20k semantic segmentation task. I.e. with a setting of 50 for the former and 5 for the latter, the pretraining is run for 50 epochs in total, testing after epoch 5, 10, 15, ...
- you can optionally add the `--stack` flag. This will download the pretrained checkpoint from the MAE paper, and then attempt to continue pretraining with the chose mask-method

For the first pretraining test, I would use the given settings: the four_channels mask method and a coverage ratio of 15%. The latter is an arbitrary choice that feels reasonable to me. The former seems like the best bet for now, as `segments` doesn't include the mask information, and `preencoder` throws away part of the encoder after pretraining.

According to our calculations, pretraining for 50 epochs with evaluation on every 5th should take roughly 4 days on 8xV100 GPUs, though pretraining on the original `patches` method for comparisons should be faster by a factor of 2 or 3. I'm not sure what a good number of pretraining epochs would be here, perhaps we should also lower this...

# Downloading additional datasets and checkpoints
The evaluation task uses the ADE20k dataset. This is expected to be in `segmenter/data/ade20k`. If this folder does not exist, it will be automatically downloaded and prepared when running the benchmark script for the first time. If you want to download _just_ the dataset, you can start the benchmark script like this:

`python benchmark.py --in1k-dir=whatever`

The dataset will be downloaded into `segmenter/data/ade20k` and prepared for processing. An example run would look like this:

```
>>> python benchmark.py --in1k-dir=whatever
Downloading segmenter/data/ade20k/downloads/ADEChallengeData2016.zip from http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip...
944710KB [01:23, 11273.07KB/s]                                                                                                               
Downloading segmenter/data/ade20k/downloads/release_test.zip from http://data.csail.mit.edu/places/ADEchallenge/release_test.zip...
100%|██████████████████████████████████████████████████████████████████████████| 206856/206856 [00:20<00:00, 10207.88KB/s]
```

When training with the `--stack` option, the pretrained model checkpoint from the MAE paper is required. This will also be downloaded automatically if not already present, if `--stack` is specified. If you want to only download the checkpoint, you can start the benchmark script like this:

`python benchmark.py --stack --in1k-dir=whatever`

The checkpoint will be downloaded into the repository root. An example run would look like this:

```
>>> python benchmark.py --stack --in1k-dir=whatever
Downloading pretrained checkpoint...
--2024-07-17 13:24:12--  https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
Resolving dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)... 2600:9000:2684:f800:13:6e38:acc0:93a1, 2600:9000:2684:a00:13:6e38:acc0:93a1, 2600:9000:2684:f600:13:6e38:acc0:93a1, ...
Connecting to dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)|2600:9000:2684:f800:13:6e38:acc0:93a1|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1213314565 (1.1G) [binary/octet-stream]
Saving to: ‘mae_pretrain_vit_large.pth’

mae_pretrain_vit_large.pth          100%[================================================================>]   1.13G  16.4MB/s    in 82s     

2024-07-17 13:25:35 (14.1 MB/s) - ‘mae_pretrain_vit_large.pth’ saved [1213314565/1213314565]

```
