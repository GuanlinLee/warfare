import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image
import argparse
import sys
import os
import yaml
import random
import numpy as np
import time
import gc

sys.path.append("../")

from DiffPure.utils import str2bool, dict2namespace
import utils
from models import Watermark_Decoder, StegaStamp
from DiffPure.runners.diffpure_ddpm import Diffusion
from DiffPure.runners.diffpure_guided import GuidedDiffusion
from DiffPure.runners.diffpure_sde import RevGuidedDiffusion
from DiffPure.runners.diffpure_ode import OdeGuidedDiffusion
from DiffPure.runners.diffpure_ldsde import LDGuidedDiffusion


class CaseStudy_Dataset(VisionDataset):
    def __init__(self, root, watermark_length, transform=None, target_transform=None):
        super(CaseStudy_Dataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.watermarked_path = os.path.join(root, 'watermark_%dbit' % (watermark_length))
        self.file_list = ['{}.pth'.format(i) for i in range(10000)]
        self.type = type

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        watermarked_img_name = os.path.join(self.watermarked_path, self.file_list[idx])
        watermarked_img = torch.load(watermarked_img_name)
        return watermarked_img, watermarked_img

class SDE_Model(nn.Module):
	def __init__(self, args, config):
		super().__init__()
		self.args = args
		# diffusion model
		print(f'diffusion_type: {args.diffusion_type}')
		if args.diffusion_type == 'ddpm-guided':
			self.runner = GuidedDiffusion(args, config, device=config.device)
		elif args.diffusion_type == 'sde':
			self.runner = RevGuidedDiffusion(args, config, device=config.device)
		elif args.diffusion_type == 'ode':
			self.runner = OdeGuidedDiffusion(args, config, device=config.device)
		elif args.diffusion_type == 'ldsde':
			self.runner = LDGuidedDiffusion(args, config, device=config.device)
		elif args.diffusion_type == 'ddpm':
			self.runner = Diffusion(args, config, device=config.device)
		else:
			raise NotImplementedError('unknown diffusion type')

	def forward(self, x):
		# imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
		if 'imagenet' in self.args.dataset:
			x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
		x_re = self.runner.image_editing_sample((x - 0.5) * 2)
		x_re = torch.clamp(x_re, -1.0, 1.0)
		if 'imagenet' in self.args.dataset:
			x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)
		out = (x_re + 1) * 0.5

		return out
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # case study
    parser.add_argument('--csm', '--case_study_model', type=str, default='EDM', choices=['WGAN', 'DDIM', 'EDM'])
    # diffusion models
    parser.add_argument('--config', type=str, default='celeba-nothq.yml', help='Path to the config file', choices=['cifar10.yml', 'bedroom.yml', 'celeba-nothq.yml'])
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")

    parser.add_argument('--sample_step', type=int, default=30, help='Total sampling steps')#try to increase sampling steps, making better results
    parser.add_argument('--t', type=int, default=150, help='Sampling noise scale')#try to increase sampling noise scale, the results can be worse if sample step is not large enough

    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')#TODO: this can be further studied
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')#TODO: this can be further studied

    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='score_sde', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')
    # debug
    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-2, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='custom')

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)
    parser.add_argument('--gpus', type=str, default='0')

    #Evaluation Settings
    parser.add_argument('--dataset', default='lsun-bedroom', type=str,
                    help='which dataset used to train', choices=['cifar10', 'lsun-bedroom', 'celeba'])
    parser.add_argument('--seed_data', type=int,
                        default=0, help='random seed')
    parser.add_argument('--save', default='lsun-bedroom', type=str,
                        help='model save name', choices=['cifar-10', 'lsun-bedroom', 'celeba'])
    parser.add_argument('--expn', default='lsun-bedroom', type=str,
                        help='exp name', choices=['cifar-10', 'lsun-bedroom', 'celeba'])#conflict

    # AE settings
    parser.add_argument('--watermark_type', default='text', type=str,
                        help='watermark type')
    parser.add_argument('--watermark_size', default=64, type=int)
    parser.add_argument('--watermark_path', default='watermark.png', type=str)
    parser.add_argument('--watermark_text', default='0100010001000010111010111111110011101000001111101101010110000000', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # parse config file
    with open(os.path.join('../DiffPure/configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device('cuda')
    new_config.device = device
    args.save = args.dataset + '/' + args.expn + '/' + args.save
    args.savepath = os.path.join('../exp', 'diffpure', args.dataset)

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


args, config = parse_args_and_config()

model = SDE_Model(args, config)


def generate():
    batchsize = 8
    train_set = CaseStudy_Dataset(root='./data/{}'.format(args.csm), watermark_length=args.watermark_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize,
                                                    shuffle=False, drop_last=False, num_workers=0)
    print("loading model")
    model = SDE_Model(args, config)

    model = model.eval().to(config.device)
    print("device is: ", config.device)

    if args.csm == 'EDM':
        watermark_decoder = StegaStamp.StegaStampDecoder()
        watermark_decoder.load_state_dict(torch.load('./data/weights/EDM/stegastamp_64_decoder.pth'))
    else:
        watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()
        watermark_decoder.load_state_dict(torch.load('../' + args.dataset + '/' + args.dataset + '/' + args.dataset + 'owner_Decoder_%dbit.pth'%(args.watermark_size)))

    watermark_decoder.eval().to(config.device)

    watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
                                    args.watermark_size, args.watermark_text)

    watermark = torch.Tensor(watermark).to(config.device)

    cwd = os.getcwd()
    print('cwd is: ', cwd)
    if not os.path.exists(cwd + '/data/{}/clean_{}bit'.format(args.csm, args.watermark_size)):
        os.makedirs(cwd + '/data/{}/clean_{}bit'.format(args.csm, args.watermark_size))

    correct = 0
    total = 0
    T1 = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            T2 = time.time()
            print("round ", i, "/", len(train_loader), "   current run time: ", (T2 - T1),"s")
        
            images = images.to(config.device)
            if args.watermark_type == 'text':
                watermark_ = watermark.tile((images.shape[0], 1)).to(config.device)
            else:
                watermark_ = watermark.tile((images.shape[0], 1, 1, 1)).to(config.device)

            wm_image = images.float()

            clean_image = model(wm_image)

            bs = images.size(0)
            clean_image = clean_image[-bs:]

            save_image(torch.cat((wm_image, clean_image), dim=0), cwd + '/data/{}/diffusion_test.png'.format(args.csm), normalize=True)

            clean_image = clean_image.detach()

            for j in range(bs):
                pic_id = j + i * batchsize
                torch.save(clean_image[j].cpu(),cwd + '/data/{}/clean_{}bit/{}.pth'.format(args.csm, args.watermark_size, pic_id))

            decoded = watermark_decoder(clean_image)
            if args.csm == 'EDM':
                decoded = (decoded > 0).long()
            else:
                decoded = torch.round(decoded)    
            total += batchsize
            correct += (decoded == watermark_).sum().item() / decoded.shape[1]

            print("test accuracy of ", total, " samples:  ", (100 * correct / total))

            del model
            gc.collect()
            model = SDE_Model(args, config)
            model = model.eval().to(config.device)

    print("finish")

def test_watermark():
    batchsize = 512
    train_set = CaseStudy_Dataset(root='./data/{}'.format(args.csm), watermark_length=64)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize,
                                                    shuffle=False, drop_last=False, num_workers=0)
    watermark_decoder = StegaStamp.StegaStampDecoder()
    watermark_decoder.load_state_dict(torch.load('./data/weights/EDM/stegastamp_64_decoder.pth'))
 
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.float()
            decoded = watermark_decoder(images)
            decoded = (decoded > 0).long()
            print(decoded.size())
            print(decoded)
            result = decoded.sum(dim=0) / decoded.size(0)
            print(result.size())
            print(result)
            break 
    """
        010001000100001011
        101011111111001110
        100000111110110101
        0110000000

        0100010001000010111010111111110011101000001111101101010110000000
    """
if __name__ == '__main__':
    generate()
    #test_watermark()

