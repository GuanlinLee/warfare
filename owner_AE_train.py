from models import OwnerEncoder, Watermark_Decoder
import torch
import dataloader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import torch.optim as optim
from logging import getLogger
import os
import argparse
import utils
import models
import lpips

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=3, help='GPU to use [default: GPU 0]')
parser.add_argument('--dataset', default='celeba', type=str,
					help='which dataset used to train', choices=['cifar10', 'lsun-bedroom', 'celeba'])#change save and exp at the same time
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
					metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--seed', type=int,
					default=0, help='random seed')

parser.add_argument('--save', default='celeba', type=str,
					help='model save name', choices=['cifar-10', 'lsun-bedroom', 'celeba'])
parser.add_argument('--exp', default='celeba', type=str,
					help='exp name',choices=['cifar-10', 'lsun-bedroom', 'celeba'])
				

#AE settings
parser.add_argument('--watermark_type', default='text', type=str,
					help='watermark type')
parser.add_argument('--watermark_size', default=48, type=int)
parser.add_argument('--watermark_path', default='watermark.png', type=str)
parser.add_argument('--watermark_text', default='100010001000100010001000100010001000100010001000', type=str,
                    help='watermark text for testing, not for training, during the training, the watermark is generated randomly')

#StegaStamp settings

parser.add_argument('--l2_loss_scale', type=float, default=2.)
parser.add_argument('--l2_loss_ramp', type=int, default=15000)
parser.add_argument('--l2_edge_gain', type=float, default=10.0)
parser.add_argument('--l2_edge_ramp', type=int, default=10000)
parser.add_argument('--l2_edge_delay', type=int, default=80000)
parser.add_argument('--lpips_loss_scale', type=float, default=1.5)
parser.add_argument('--lpips_loss_ramp', type=int, default=15000)
parser.add_argument('--secret_loss_scale', type=float, default=1.5)
parser.add_argument('--secret_loss_ramp', type=int, default=1)
parser.add_argument('--G_loss_scale', type=float, default=0.5)
parser.add_argument('--G_loss_ramp', type=int, default=15000)
parser.add_argument('--y_scale', type=float, default=1.0)
parser.add_argument('--u_scale', type=float, default=100.0)
parser.add_argument('--v_scale', type=float, default=100.0)
parser.add_argument('--no_gan', action='store_true')
parser.add_argument('--rnd_trans', type=float, default=.1)
parser.add_argument('--rnd_bri', type=float, default=.3)
parser.add_argument('--rnd_noise', type=float, default=.02)
parser.add_argument('--rnd_sat', type=float, default=1.0)
parser.add_argument('--rnd_hue', type=float, default=.1)
parser.add_argument('--contrast_low', type=float, default=.5)
parser.add_argument('--contrast_high', type=float, default=1.5)
parser.add_argument('--jpeg_quality', type=float, default=50)
parser.add_argument('--no_jpeg', action='store_true')
parser.add_argument('--rnd_trans_ramp', type=int, default=10000)
parser.add_argument('--rnd_bri_ramp', type=int, default=1000)
parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
parser.add_argument('--contrast_ramp', type=int, default=1000)
parser.add_argument('--jpeg_quality_ramp', type=float, default=1000)
parser.add_argument('--no_im_loss_steps', help="Train without image loss for first x steps", type=int, default=1500)

#Data parallel settings
parser.add_argument('--dp', action='store_true', help='enable pytorch data parallel')

args = parser.parse_args()



if args.dp == False:
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
	device_ids = [0,1]

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

if not os.path.exists(args.save):
	os.mkdir(args.save)


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
	args.batch_size = 32
	args.epochs = 500
	args.l2_loss_scale = 10.0
	args.lpips_loss_scale = 2
	args.G_loss_scale = 1.0
	args.secret_loss_scale = 2.0

data = dataloader.Data(args.dataset, './data')
trainset, testset = data.data_loader(transform, transform_test)

learning_rate=args.lr
epochs=args.epochs
batch_size=args.batch_size
torch.backends.cudnn.benchmark = True

print("save is", args.save)

if args.dataset == 'cifar10':
	new_dataset = utils.split_dataset(trainset, 25000, args.save, None, None)
elif args.dataset == 'lsun-bedroom':
	new_dataset, _ = utils.split_dataset_lsun(trainset, 100000, args.save, None, None)
elif args.dataset == 'celeba':
	new_dataset, _ = utils.split_dataset_celeba(trainset, 100000, args.save, None, None)

train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size,
                                                shuffle=True, drop_last=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, drop_last=False, num_workers=0)

watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
                                 args.watermark_size, args.watermark_text)


watermark = torch.Tensor(watermark).cuda()


#load model
autoencoder = OwnerEncoder.Autoencoder(args.watermark_type, args.watermark_size, resolution=resolution).cuda()#resolution changed to a list
watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()
discriminator = models.utils.Discriminator().cuda()

#wrap dp model
if args.dp:
	autoencoder = torch.nn.DataParallel(autoencoder, device_ids=device_ids)
	watermark_decoder = torch.nn.DataParallel(watermark_decoder, device_ids=device_ids)
	discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)
	d_vars = discriminator.module.parameters()
	g_vars = [{'params': autoencoder.module.parameters()},
	          {'params': watermark_decoder.module.parameters()}]
	num_gpus = len(device_ids)
else:
	d_vars = discriminator.parameters()
	g_vars = [{'params': autoencoder.parameters()},
	          {'params': watermark_decoder.parameters()}]

#load optimizer

optimize_loss = optim.Adam(g_vars, lr=args.lr)
optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

#load loss function
criterion_decoder = torch.nn.BCELoss().cuda()
criterion_ae = torch.nn.MSELoss().cuda()
lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).cuda()

global_step = 0
#train
for epoch in range(epochs):
	autoencoder.train()
	watermark_decoder.train()
	for i, (images, labels) in enumerate(train_loader):

		no_im_loss = global_step < args.no_im_loss_steps
		l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
		lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
		secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
		                        args.secret_loss_scale)
		G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
		l2_edge_gain = 0
		if global_step > args.l2_edge_delay:
			l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
			                   args.l2_edge_gain)

		rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
		rnd_tran = np.random.uniform() * rnd_tran
		global_step += 1

		loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
		yuv_scales = [args.y_scale, args.u_scale, args.v_scale]

		images = images.cuda()
		labels = labels.cuda()
		if args.watermark_type == 'text':
			watermark_ = torch.randint(0, 2, (images.size(0), args.watermark_size)).cuda()
		else:
			watermark_ = watermark.tile((images.shape[0], 1, 1, 1)).cuda()
		#train autoencoder
		residual = autoencoder(images, watermark_)
		encoded_images = images + residual

		#test the quality of encoded image
		if i % 100 == 0:
			if not os.path.exists(args.dataset + '/test_image'):
				os.makedirs(args.dataset + '/test_image')
			save_image(torch.cat((images, encoded_images), dim=0), os.path.join(args.dataset, 'test_image/', 'watermark_test.png'))

		distance = ((images - encoded_images)**2).sum()
		distance = distance.item()

		D_output_real, _ = discriminator(images)
		D_output_fake, D_heatmap = discriminator(encoded_images)
		transformed_image = models.utils.transform_net(encoded_images, args, global_step, resolution=resolution)

		decoded_secret = watermark_decoder(transformed_image)

		loss_lpips = torch.mean(lpips_loss_fn(encoded_images * 2.0 - 1.0, images * 2.0 - 1.0))

		watermark_ = watermark_.to(torch.float32)

		loss_bce = criterion_decoder(decoded_secret, watermark_)
		loss_image = models.utils.image_loss_fn(images, encoded_images, l2_edge_gain, yuv_scales)

		D_loss = D_output_real - D_output_fake
		G_loss = D_output_fake
		loss = torch.tensor(loss_scales[0]).view(1).cuda() * loss_image + \
		       torch.tensor(loss_scales[1]).view(1).cuda() * loss_lpips + \
		       torch.tensor(loss_scales[2]).view(1).cuda() * loss_bce
		if not args.no_gan:
			if args.dp:
				loss += torch.tensor(loss_scales[3]).view(1).cuda() * G_loss.mean()
			else:
				loss += torch.tensor(loss_scales[3]).view(1).cuda() * G_loss

		if no_im_loss:
			optimize_secret_loss.zero_grad()
			loss_bce.backward()
			#clip gradient
			if args.dp:
				torch.nn.utils.clip_grad_norm_(watermark_decoder.module.parameters(), 0.25)
				torch.nn.utils.clip_grad_norm_(autoencoder.module.parameters(), 0.25)
			else:
				torch.nn.utils.clip_grad_norm_(watermark_decoder.parameters(), 0.25)
				torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 0.25)
			optimize_secret_loss.step()
		else:
			optimize_loss.zero_grad()
			loss.backward()
			#clip gradient
			if args.dp:
				torch.nn.utils.clip_grad_norm_(watermark_decoder.module.parameters(), 0.25)
				torch.nn.utils.clip_grad_norm_(autoencoder.module.parameters(), 0.25)
			else:
				torch.nn.utils.clip_grad_norm_(watermark_decoder.parameters(), 0.25)
				torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 0.25)
			optimize_loss.step()
			if not args.no_gan:
				optimize_dis.zero_grad()
				# clip gradient
				if args.dp:
					torch.nn.utils.clip_grad_norm_(discriminator.module.parameters(), 0.01)
				else:
					torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.01)
				optimize_dis.step()

		if i % 100 == 0:
			logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
						.format(epoch + 1, epochs, global_step, len(train_loader) * epochs, loss.item()))
			print("the distance is ", distance)


	#test
	autoencoder.eval()
	watermark_decoder.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = images.cuda()
			labels = labels.cuda()
			if args.watermark_type == 'text':
				watermark_ = watermark.tile((images.shape[0], 1)).cuda()
			else:
				watermark_ = watermark.tile((images.shape[0], 1, 1, 1)).cuda()
			outputs = autoencoder(images, watermark_)
			outputs = watermark_decoder(outputs)
			outputs = outputs.detach()
			outputs = torch.round(outputs)
			total += labels.size(0)
			if args.watermark_type == 'text':
				correct += (outputs == watermark_).sum().item() / outputs.shape[1]
			else:
				pass
		logger.info('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
		if (epoch + 1) % 10 == 0:
			if args.watermark_type == 'text':
				if args.dp:
					torch.save(autoencoder.module.state_dict(), args.save + 'owner_AE_%dbit.pth'%(args.watermark_size))
					torch.save(watermark_decoder.module.state_dict(), args.save + 'owner_Decoder_%dbit.pth'%(args.watermark_size))
				else:
					torch.save(autoencoder.state_dict(), args.save + 'owner_AE_%dbit.pth'%(args.watermark_size))
					torch.save(watermark_decoder.state_dict(), args.save + 'owner_Decoder_%dbit.pth'%(args.watermark_size))




