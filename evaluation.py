from models import Adversary_AE, Watermark_Decoder, OwnerEncoder
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
parser.add_argument('--dataset', default='celeba', type=str,
					help='which dataset used to train', choices=['cifar10', 'lsun-bedroom', 'celeba'])
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
					default=200, help='w_G in paper, hyper parameter between adv target & image quality')
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
parser.add_argument('--watermark_size', default=32, type=int)
parser.add_argument('--watermark_path', default='watermark.png', type=str)
parser.add_argument('--watermark_text', default='10001000100010001000100010001000', type=str)
parser.add_argument('--train_size', default=10000, type=int,
					help='number of adversary samples', choices=[25000, 20000, 15000, 10000, 5000, 2000, 1000])


#Threat Model settings
parser.add_argument('--target', default='remove', type=str,
					help='threat model type', choices=['remove', 'forge'])
parser.add_argument('--epoch', default=100, type=int,
					help='which epoch model to use')

#EMA settings
parser.add_argument('--model_ema', action='store_true', help='enable EMA of model parameters')

#fewshot settings
parser.add_argument('--fs', action='store_true', help='enable fewshot settings')
parser.add_argument('--watermark_text_fs', default='11100011101010101000010000001011', type=str)
parser.add_argument('--train_size_fewshot', default=10, type=int,
					help='number of adversary samples')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
utils.setup_seed(args.seed)


logger = getLogger()
if not os.path.exists(args.dataset + '/' + args.exp):
	os.makedirs(args.dataset + '/' + args.exp)
logger = utils.create_logger(
	os.path.join(args.dataset + '/' + args.exp + '/', args.exp + ".log"), rank=0
)
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)
args.save = args.dataset + '/' + args.exp + '/' + args.save

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
elif args.dataset == 'celeba':
	transform=transforms.Compose([transforms.Resize((64,64)),#default resolution is 178x218
							  transforms.ToTensor(),
							  ])
	transform_test=transforms.Compose([transforms.Resize((64,64)),
								   transforms.ToTensor(),
								   ])
	resolution = [64,64]

data = dataloader.Data(args.dataset, './data')
trainset, testset = data.data_loader(transform, transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, drop_last=False, num_workers=0)

if args.fs == True:
	watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
                                 args.watermark_size, args.watermark_text_fs)
else:
	watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
									args.watermark_size, args.watermark_text)

watermark = torch.Tensor(watermark).cuda()

#load model
if args.dataset == 'cifar10':
	autoencoder_adversary = Adversary_AE.ResnetGenerator(3, 3).cuda()
else:
	autoencoder_adversary = Adversary_AE.ResnetGenerator_big(3, 3).cuda()

autoencoder = OwnerEncoder.Autoencoder(args.watermark_type, args.watermark_size, resolution).cuda()
watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()

#load models
if args.watermark_type == 'text':
	if args.model_ema == True:
		autoencoder_adversary.load_state_dict(torch.load(args.save + 'Adversary_AE_%dbit_%d_%s_%d_w=%d_ema.pth'%(args.watermark_size, args.train_size, args.target, args.epoch, args.w_errG)))
	elif args.fs == True:
		autoencoder_adversary.load_state_dict(torch.load(args.save + 'Adversary_AE_%dbit_%d_%s_%d_w=%d_fewshot.pth'%(args.watermark_size, args.train_size_fewshot, args.target, args.epoch, args.w_errG)))                   
	else:
		print("load name is :", 'Adversary_AE_%dbit_%d_%s_%d_w=%d.pth'%(args.watermark_size, args.train_size, args.target, args.epoch, args.w_errG))
		autoencoder_adversary.load_state_dict(torch.load(args.save + 'Adversary_AE_%dbit_%d_%s_%d_w=%d.pth'%(args.watermark_size, args.train_size, args.target, args.epoch, args.w_errG)))
	autoencoder.load_state_dict(torch.load(args.save + 'owner_AE_%dbit.pth'%(args.watermark_size)))
	watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_%dbit.pth'%(args.watermark_size)))

elif args.watermark_type == 'image':
	autoencoder_adversary.load_state_dict(torch.load(args.save + 'Adversary_AE_img_%s_%d.pth'%(args.target, args.epoch)))
	autoencoder.load_state_dict(torch.load(args.save + 'owner_AE_img.pth'))
	watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_img.pth'))

#test
autoencoder_adversary.eval()
autoencoder.eval()
watermark_decoder.eval()
 

test_image_path_clean = args.dataset + '/test_clean/'
test_image_path_watermarked = args.dataset + '/test_watermarked/'
test_image_path_remove = args.dataset + '/test_remove/'
test_image_path_forge = args.dataset + '/test_forge/'

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

if not os.path.exists(test_image_path_clean):
	os.mkdir(test_image_path_clean)

if not os.path.exists(test_image_path_watermarked):
	os.mkdir(test_image_path_watermarked)

if not os.path.exists(test_image_path_remove):
	os.mkdir(test_image_path_remove)

if not os.path.exists(test_image_path_forge):
	os.mkdir(test_image_path_forge)



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

	for images, labels in test_loader:
		
		images = images.cuda()

		img_processed = preprocess(images).cuda()
		image_features = clip_model.encode_image(img_processed)
		image_features /= image_features.norm(dim=-1, keepdim=True)

		if args.watermark_type == 'text':
			watermark_ = watermark.tile((images.shape[0], 1)).cuda()
		else:
			watermark_ = watermark.tile((images.shape[0], 1, 1, 1)).cuda()
		if args.target == 'remove':
			watermarked = autoencoder(images, watermark_) + images
			outputs_img = autoencoder_adversary(watermarked)

			outputs = watermark_decoder(outputs_img)
			outputs = outputs.detach()
			outputs = torch.round(outputs)
			total += labels.size(0)
			if args.watermark_type == 'text':
				correct += (outputs == watermark_).sum().item() / outputs.shape[1]
			else:
				pass
			for img_c, img in zip(images, outputs_img):
				save_image(img, os.path.join(test_image_path_remove, 'remove_%d.png'%adv_i))
				adv_i += 1
				psnr, ssim = compute_psnr_ssim(img_c, img)
				adv_ssim += ssim
				adv_psnr += psnr

			watermarked_img = preprocess(watermarked).cuda()
			watermarked_features = clip_model.encode_image(watermarked_img)
			watermarked_features /= watermarked_features.norm(dim=-1, keepdim=True)

			outputs_img_p = preprocess(outputs_img).cuda()
			outputs_features = clip_model.encode_image(outputs_img_p)
			outputs_features /= outputs_features.norm(dim=-1, keepdim=True)

			clip_sim_watermark += (image_features * watermarked_features).sum(dim=-1).sum().item()
			clip_sim_adv += (image_features * outputs_features).sum(dim=-1).sum().item()

		elif args.target == 'forge':

			forged = autoencoder_adversary(images)
			outputs = watermark_decoder(forged)
			outputs = outputs.detach()
			outputs = torch.round(outputs)
			total += labels.size(0)
			if args.watermark_type == 'text':
				correct += (outputs == watermark_).sum().item() / outputs.shape[1]
			else:
				pass
			for img_c, img in zip(images, forged):
				save_image(img, os.path.join(test_image_path_forge, 'forge_%d.png'%adv_i))
				adv_i += 1
				psnr, ssim = compute_psnr_ssim(img_c, img)
				adv_ssim += ssim
				adv_psnr += psnr

			forged_img = preprocess(forged).cuda()
			forged_features = clip_model.encode_image(forged_img)
			forged_features /= forged_features.norm(dim=-1, keepdim=True)

			clip_sim_adv += (image_features * forged_features).sum(dim=-1).sum().item()

		logger.info('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

		for img in images:
			save_image(img, os.path.join(test_image_path_clean, 'img_%d.png'%real_i))
			real_i += 1
		if args.target == 'remove':
			for img_c, img in zip(images, watermarked):
				save_image(img, os.path.join(test_image_path_watermarked, 'img_%d.png'%fake_i))
				fake_i += 1
				psnr, ssim = compute_psnr_ssim(img_c, img)
				watermark_ssim += ssim
				watermark_psnr += psnr

	logger.info('PSNR of watermark: %.4f' % (watermark_psnr / testset.__len__()))
	logger.info('SSIM of watermark: %.4f' % (watermark_ssim / testset.__len__()))
	logger.info(f'PSNR of {args.target}: %.4f' % (adv_psnr / testset.__len__()))
	logger.info(f'SSIM of {args.target}: %.4f' % (adv_ssim / testset.__len__()))

	if args.target == 'remove':
		logger.info('CLIP similarity of watermark: %.4f' % (clip_sim_watermark / testset.__len__()))
	logger.info(f'CLIP similarity of {args.target}: %.4f' % (clip_sim_adv / testset.__len__()))




