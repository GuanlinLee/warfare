import sys
sys.path.append("../")
from models import Adversary_AE, Watermark_Decoder, StegaStamp
import torch
import dataloader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from logging import getLogger
import os
import argparse
import utils
import clip


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--csm', '--case_study_model', type=str, default='EDM', choices=['WGAN', 'DDIM', 'EDM'])
parser.add_argument('--dataset', type=str,
					help='which dataset used to train', choices=['cifar10', 'lsun-bedroom', 'celeba', 'ffhq'])
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.000002, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--w_errG', type=float,
					default=250, help='w_G in paper, hyper parameter between adv target & image quality')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='wd')

parser.add_argument('--seed', type=int,
					default=0, help='random seed')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')


parser.add_argument('--save', default='celeba', type=str,
					help='model save name', choices=['cifar10', 'lsun-bedroom', 'celeba'])
parser.add_argument('--exp', default='celeba', type=str,
					help='exp name', choices=['cifar10', 'lsun-bedroom', 'celeba'])

#AE settings
parser.add_argument('--watermark_type', default='text', type=str,
					help='watermark type')
parser.add_argument('--watermark_size', default=64, type=int)
parser.add_argument('--watermark_path', default='watermark.png', type=str)
parser.add_argument('--watermark_text', default='0100010001000010111010111111110011101000001111101101010110000000', type=str)
parser.add_argument('--train_size', default=10000, type=int,
					help='number of adversary samples', choices=[25000, 20000, 15000, 10000, 5000, 2000, 1000])
#EDM: 0100010001000010111010111111110011101000001111101101010110000000

#Threat Model settings
parser.add_argument('--target', default='remove', type=str,
					help='threat model type', choices=['remove', 'forge'])
parser.add_argument('--epoch', default=500, type=int,
					help='which epoch model to use')

#EMA settings
parser.add_argument('--model_ema', action='store_true', help='enable EMA of model parameters')

args = parser.parse_args()

if args.dataset == None:
	if args.csm == 'EDM':
		args.dataset = 'ffhq'
	else:
		args.dataset = 'celeba'

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
utils.setup_seed(args.seed)


logger = getLogger()
if not os.path.exists('./data/log'):
	os.makedirs('./data/log')
logger = utils.create_logger(
	os.path.join('./data/log', args.exp + ".log"), rank=0
)
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)

args.save = '../' + args.dataset + '/' + args.exp + '/' + args.save

clip_model, p = clip.load('ViT-B/32', 'cuda')
clip_model.eval()
preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224),
                                            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
											torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))])

wd=args.wd
learning_rate=args.lr
epochs=args.epochs
batch_size=args.batch_size
torch.backends.cudnn.benchmark = True

if args.dataset == 'cifar10':
	transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
							  transforms.RandomHorizontalFlip(),
							  transforms.ToTensor(),
							  ])
	transform_test=transforms.Compose([transforms.Resize((32,32)),
								   transforms.ToTensor(),
								   ])
	resolution = [32,32]
elif args.dataset == 'lsun-bedroom':
	transform=transforms.Compose([transforms.Resize((256,256)),
							  #transforms.RandomHorizontalFlip(),
							  transforms.ToTensor(),
							  ])
	transform_test=transforms.Compose([transforms.Resize((256,256)),
								   transforms.ToTensor(),
								   ])
	resolution = [256,256]
	args.batch_size = 32
elif args.dataset == 'celeba' or args.dataset == 'ffhq':
	transform=transforms.Compose([transforms.Resize((64,64)),#default resolution is 178x218
							  transforms.ToTensor(),
							  ])
	transform_test=transforms.Compose([transforms.Resize((64,64)),
								   transforms.ToTensor(),
								   ])
	resolution = [64,64]

"""
data = dataloader.Data(args.dataset, '../data')
trainset, testset = data.data_loader(transform, transform_test)

train_indices = random.sample(range(len(testset)), 1000)

testset = torch.utils.data.Subset(testset, train_indices)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, drop_last=False, num_workers=0)
"""
if args.target == 'remove':
	testset = torchvision.datasets.ImageFolder(root='./data/{}/watermark_{}bit_eval'.format(args.csm, args.watermark_size),
												transform=transform_test)
	indices = range(10000)
	testset = torch.utils.data.Subset(testset, indices)
elif args.target == 'forge':
	data = dataloader.Data(args.dataset, '../data')
	_, testset = data.data_loader(transform, transform_test)

	print(len(testset))

	#indices = random.sample(range(len(testset)), 10000)
	indices = range(10000)
	testset = torch.utils.data.Subset(testset, indices)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
												shuffle=False, drop_last=False, num_workers=0)

watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
                                 args.watermark_size, args.watermark_text)

watermark = torch.Tensor(watermark).cuda()

#load model
if args.dataset == 'cifar10':
	autoencoder_adversary = Adversary_AE.ResnetGenerator(3, 3).cuda()
else:
	autoencoder_adversary = Adversary_AE.ResnetGenerator_big(3, 3).cuda()

if args.csm == 'EDM':
	watermark_decoder = StegaStamp.StegaStampDecoder().cuda()
	watermark_decoder.load_state_dict(torch.load('./data/weights/EDM/stegastamp_64_decoder.pth'))
else:
	watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()
	watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_%dbit.pth'%(args.watermark_size)))

#load models
if args.watermark_type == 'text':
	autoencoder_adversary.load_state_dict(torch.load('./data/weights/%s/'%(args.csm) + '%s_AE_%dbit_%d_%s_%d_w=%d.pth'%(args.csm, args.watermark_size, args.train_size, args.target, args.epoch, args.w_errG)))
	print("load model name is: ", './data/weights/%s/'%(args.csm) + '%s_AE_%dbit_%d_%s_%d_w=%d.pth'%(args.csm, args.watermark_size, args.train_size, args.target, args.epoch, args.w_errG))

#test
autoencoder_adversary.eval()
watermark_decoder.eval()
 

test_image_path_clean = './data' + '/fid_test' + '/test_clean/'
test_image_path_watermarked = './data' + '/fid_test' + '/test_watermarked/'
test_image_path_remove = './data' + '/fid_test' + '/test_remove/'
test_image_path_forge = './data' + '/fid_test' + '/test_forge/'

if not os.path.exists('./data/fid_test'):
	os.mkdir('./data/fid_test')

if not os.path.exists(test_image_path_clean):
	os.mkdir(test_image_path_clean)

if not os.path.exists(test_image_path_watermarked):
	os.mkdir(test_image_path_watermarked)

if not os.path.exists(test_image_path_remove):
	os.mkdir(test_image_path_remove)

if not os.path.exists(test_image_path_forge):
	os.mkdir(test_image_path_forge)

def compute_psnr_ssim(x_clean, x_fake):
    x1 = x_clean.reshape(3,x_clean.shape[1],x_clean.shape[2]).detach().cpu().numpy()
    x2 = x_fake.reshape(3,x_clean.shape[1],x_clean.shape[2]).detach().cpu().numpy()
    x1 = np.transpose(x1, (1,2,0)) * 255.0
    x2 = np.transpose(x2, (1,2,0)) * 255.0
    x1 = np.clip(x1, 0.0, 255.0)
    x2 = np.clip(x2, 0.0, 255.0)
    ssims = []
    for i in range(3):
        ssims.append(utils.ssim_(x1[:,:,i], x2[:,:,i]))
    ssim_dist = np.array(ssims).mean()
    psnr_dist = utils.psnr(x1, x2)
    return psnr_dist, ssim_dist

with torch.no_grad():
	correct = 0
	total = 0
	real_i = 0
	fake_i = 0
	adv_i = 0
	watermark_ssim = 0.0
	watermark_psnr = 0.0
	adv_ssim = 0.0
	adv_psnr = 0.0

	clip_sim_watermark = 0.0
	clip_sim_adv = 0.0

	quality_rank = []
	for images, labels in test_loader:
		
		images = images.cuda().float()

		if args.watermark_type == 'text':
			watermark_ = watermark.tile((images.shape[0], 1)).cuda()
		else:
			watermark_ = watermark.tile((images.shape[0], 1, 1, 1)).cuda()
		if args.target == 'remove':
			outputs_img = autoencoder_adversary(images)
			outputs = watermark_decoder(outputs_img)
			outputs = outputs.detach()
			if args.csm == 'EDM':
				outputs = (outputs > 0).long()
			else:
				outputs = torch.round(outputs)
			total += labels.size(0)
			
			correct += (outputs == watermark_).sum().item() / outputs.shape[1]

			for img_c, img in zip(images, outputs_img):
				snr, _ = compute_psnr_ssim(img_c, img)
				quality_rank.append(psnr)
				save_image(img, os.path.join(test_image_path_remove, 'remove_%d.png'%adv_i))
				adv_i += 1

		elif args.target == 'forge':

			forged = autoencoder_adversary(images) #+ images

			outputs = watermark_decoder(forged)
			outputs = outputs.detach()
			if args.csm == 'EDM':
				outputs = (outputs > 0).long()
			else:
				outputs = torch.round(outputs)
			total += labels.size(0)
			if args.watermark_type == 'text':
				correct += (outputs == watermark_).sum().item() / outputs.shape[1]
			else:
				pass
			for img_c, img in zip(images, forged):
				psnr, _ = compute_psnr_ssim(img_c, img)
				quality_rank.append(psnr)
				save_image(img, os.path.join(test_image_path_forge, 'forge_%d.png'%adv_i))
				adv_i += 1

		logger.info('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

		if args.target == 'forge':
			for img in images:
				save_image(img, os.path.join(test_image_path_clean, 'img_%d.png'%real_i))
				real_i += 1
		if args.target == 'remove':
			for img in images:
				save_image(img, os.path.join(test_image_path_watermarked, 'img_%d.png'%fake_i))
				fake_i += 1
	sorted_indexes = sorted(range(len(quality_rank)), key=lambda x: quality_rank[x], reverse=True)
	print(sorted_indexes[:100])
