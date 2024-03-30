import torch
import torch.nn as nn
import torch.nn.functional as F
import DiffPure
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import os
from DiffPure.utils import str2bool,dict2namespace
import yaml
import random
import numpy as np
import utils
from models import OwnerEncoder, Watermark_Decoder
from DiffPure.runners.diffpure_ddpm import Diffusion
from DiffPure.runners.diffpure_guided import GuidedDiffusion
from DiffPure.runners.diffpure_sde import RevGuidedDiffusion
from DiffPure.runners.diffpure_ode import OdeGuidedDiffusion
from DiffPure.runners.diffpure_ldsde import LDGuidedDiffusion
import dataloader
import time
import gc

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
	parser.add_argument('--dataset', default='celeba', type=str,
					help='which dataset used to train', choices=['cifar10', 'lsun-bedroom', 'celeba'])
	parser.add_argument('--seed_data', type=int,
						default=0, help='random seed')
	parser.add_argument('--save', default='celeba', type=str,
						help='model save name', choices=['cifar-10', 'lsun-bedroom', 'celeba'])
	parser.add_argument('--expn', default='celeba', type=str,
						help='exp name', choices=['cifar-10', 'lsun-bedroom', 'celeba'])#conflict

	#fewshot
	parser.add_argument('--fs', action='store_true', help='enable fewshot settings')
	# AE settings
	parser.add_argument('--watermark_type', default='text', type=str,
						help='watermark type')
	parser.add_argument('--watermark_size', default=64, type=int)
	parser.add_argument('--watermark_path', default='watermark.png', type=str)
	parser.add_argument('--watermark_text', default='1000100010001000100010001000100010001000100010001000100010001000', type=str)
	parser.add_argument('--watermark_text_fs', default='11100011101010101000010000001011', type=str)

	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	# parse config file
	with open(os.path.join('DiffPure/configs', args.config), 'r') as f:
		config = yaml.safe_load(f)
	new_config = dict2namespace(config)
	# add device
	#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

if args.dataset == 'cifar10':
	transform_test=transforms.Compose([transforms.Resize((32,32)),
								   transforms.ToTensor(),
								   ])
	resolution = [32,32]
elif args.dataset == 'lsun-bedroom':
	transform_test=transforms.Compose([transforms.Resize((256,256)),
								   transforms.ToTensor(),
								   ])
	resolution = [256,256]
elif args.dataset == 'celeba':
	transform_test=transforms.Compose([transforms.Resize((64,64)),
								   transforms.ToTensor(),
								   ])	
	resolution = [64,64]
data = dataloader.Data(args.dataset, './data')
trainset, testset = data.data_loader(transform_test, transform_test)

if args.dataset == 'cifar10':
	train_ids = torch.load(os.path.join(args.save,'train_idx.pth'))
	val_ids = torch.load(os.path.join(args.save,'val_idx.pth'))

"""
dataset settings
"""
print("splitting dataset")
if args.dataset == 'cifar10':
	new_dataset = utils.split_dataset(trainset, 25000, args.save, val_ids, train_ids)
	total_samples = 25000
elif args.dataset == 'lsun-bedroom':
	new_dataset, _ = utils.split_dataset_lsun(trainset, 100000, args.save)
	new_dataset, _ = utils.split_dataset_lsun(new_dataset, 10000, args.save)
	total_samples = 10000
elif args.dataset == 'celeba':
	new_dataset, _ = utils.split_dataset_celeba(trainset, 100000, args.save)
	new_dataset, _ = utils.split_dataset_celeba(new_dataset, 10000, args.save)
	total_samples = 10000

if args.fs == True:
	new_dataset = torch.utils.data.Subset(new_dataset, range(500))
	total_samples = 500
batchsize = 100
train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batchsize,
                                                shuffle=True, drop_last=False, num_workers=0)
print("loading model")
model = SDE_Model(args, config)

model = model.eval().to(config.device)
print(config.device)
autoencoder = OwnerEncoder.Autoencoder(args.watermark_type, args.watermark_size, resolution=resolution)
watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()

if args.watermark_type == 'text':
	autoencoder.load_state_dict(torch.load(args.save + 'owner_AE_%dbit.pth'%(args.watermark_size)))
	watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_%dbit.pth'%(args.watermark_size)))
elif args.watermark_type == 'image':
	autoencoder.load_state_dict(torch.load(args.save + 'owner_AE_img.pth'))
	watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_img.pth'))

autoencoder.eval().to(config.device)
watermark_decoder.eval().to(config.device)

if args.fs == True:
	watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
									args.watermark_size, args.watermark_text_fs)
else:	
	watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
									args.watermark_size, args.watermark_text)

watermark = torch.Tensor(watermark).to(config.device)

if args.fs == True:
	if not os.path.exists(args.save + '/fewshot_clean_%dbit'%(args.watermark_size)):
		os.makedirs(args.save + '/fewshot_clean_%dbit'%(args.watermark_size))
	if not os.path.exists(args.save + '/fewshot_watermark_%dbit'%(args.watermark_size)):
		os.makedirs(args.save + '/fewshot_watermark_%dbit'%(args.watermark_size))
else:
	if not os.path.exists(args.save + '/clean_%dbit'%(args.watermark_size)):
		os.makedirs(args.save + '/clean_%dbit'%(args.watermark_size))
	if not os.path.exists(args.save + '/watermark_%dbit'%(args.watermark_size)):
		os.makedirs(args.save + '/watermark_%dbit'%(args.watermark_size))

data_pair = {}
clean_data = []
wm_data = []
correct = 0
total = 0
T1 = time.time()

with torch.no_grad():
	for i, (images, labels) in enumerate(train_loader):
		T2 = time.time()
		print("round ", i, "/", int(total_samples / batchsize), "   current run time: ", (T2 - T1),"s")
	
		images = images.to(config.device)
		if args.watermark_type == 'text':
			watermark_ = watermark.tile((images.shape[0], 1)).to(config.device)
		else:
			watermark_ = watermark.tile((images.shape[0], 1, 1, 1)).to(config.device)
	

		wm_image = autoencoder(images, watermark_) + images #residual is needed!
		wm_image = wm_image.detach()
		if args.dataset == 'cifar10':
			wm_data.append(wm_image.cpu())
		else:
			for j in range(batchsize):
				pic_id = j + i * batchsize
				if args.fs == True:
					torch.save(wm_image[j].cpu(),args.save + '/fewshot_watermark_%dbit/%d.pth'%(args.watermark_size, pic_id))
				else:
					torch.save(wm_image[j].cpu(),args.save + '/watermark_%dbit/%d.pth'%(args.watermark_size, pic_id))

		clean_image = model(wm_image)

		clean_image = clean_image[-batchsize:] #batch in mulitple sample steps

		if not os.path.exists(args.dataset + '/test_image'):
			os.makedirs(args.dataset + '/test_image')
		save_image(torch.cat((wm_image, clean_image), dim=0), os.path.join((args.dataset + '/test_image/'), 'diffution_test_%d.png'%i))

		clean_image = clean_image.detach()
		if args.dataset == 'cifar10':
			clean_data.append(clean_image.cpu())
		else:
			for j in range(batchsize):
				pic_id = j + i * batchsize
				if args.fs == True:
					torch.save(clean_image[j].cpu(), args.save + '/fewshot_clean_%dbit/%d.pth'%(args.watermark_size, pic_id))
				else:
					torch.save(clean_image[j].cpu(), args.save + '/clean_%dbit/%d.pth'%(args.watermark_size, pic_id))

		decoded = watermark_decoder(clean_image)
	 
		decoded = torch.round(decoded)
	
		total += batchsize
		correct += (decoded == watermark_).sum().item() / decoded.shape[1]

		print("test accuracy of ", total, " samples:  ", (100 * correct / total))

		del model
		gc.collect()
		model = SDE_Model(args, config)
		model = model.eval().to(config.device)

print("finish")
if args.dataset == 'cifar10':
	wm_data = torch.cat(wm_data, dim=0)
	clean_data = torch.cat(clean_data, dim=0)
	data_pair['wm'] = wm_data
	data_pair['clean'] = clean_data
	if args.watermark_type == 'text':
		torch.save(data_pair, args.save + '/data_pair_%dbit.pt'%(args.watermark_size))
	elif args.watermark_type == 'image':
		torch.save(data_pair, args.save + '/data_pair_imgwm.pt')
