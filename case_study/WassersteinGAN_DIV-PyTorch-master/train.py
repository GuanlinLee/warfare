# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import logging

import wgandiv_pytorch.models as models
from wgandiv_pytorch import create_folder

from trainer import Trainer

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An implementation of WassersteinGAN-DIV algorithm using PyTorch framework.")
    parser.add_argument("--data", metavar="DIR", default='../data/celeba_watermarked_32bit/',
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default='celeba',
                        help="| lsun |.")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="lsun",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (default: lsun)")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                        help="Number of data loading workers. (default:8)")
    parser.add_argument("--start-iter", default=0, type=int, metavar="N",
                        help="manual iter number (useful on restarts)")
    parser.add_argument("--iters", default=100000, type=int, metavar="N",
                        help="The number of iterations is needed in the training of model. (default: 100000)")
    parser.add_argument("-b", "--batch-size", default=64, type=int, metavar="N",
                        help="mini-batch size (default: 64), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel.")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate. (default:0.0002)")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Number of training steps for discriminator per iter. (Default: 5).")
    parser.add_argument("--image-size", type=int, default=64,
                        help="The height / width of the input image to network. (default: 64).")
    parser.add_argument("--classes", default="church_outdoor",
                        help="comma separated list of classes for the lsun data set. (default: ``church_outdoor``).")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--netD", default="", type=str, metavar="PATH",
                        help="Path to latest discriminator checkpoint. (default: ````).")
    parser.add_argument("--netG", default="", type=str, metavar="PATH",
                        help="Path to latest generator checkpoint. (default: ````).")
    parser.add_argument("--manualSeed", type=int, default=1111,
                        help="Seed for initializing training. (default:1111)")
    parser.add_argument("--device", default="0",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``0``).")
    #Watermark settings   
    parser.add_argument("--watermark_loss", default=False, type=bool, help='enable BECLoss of decoded watermark')
    parser.add_argument('--watermark_type', default='text', type=str,
                        help='watermark type')
    parser.add_argument('--watermark_size', default=32, type=int)
    parser.add_argument('--watermark_path', default='watermark.png', type=str)
    parser.add_argument('--watermark_text', default='10001000100010001000100010001000', type=str)

    args = parser.parse_args()

    print("##################################################\n")
    print("Run Training Engine.\n")
    print(args)

    create_folder("output")
    create_folder("weights")

    logger.info("TrainingEngine:")
    print("\tAPI version .......... 0.1.0")
    print("\tBuild ................ 2020.12.18-1454-f636e462")

    logger.info("Creating Training Engine")
    trainer = Trainer(args)

    logger.info("Staring training model")
    trainer.run()
    print("##################################################\n")

    logger.info("All training has been completed successfully.\n")
