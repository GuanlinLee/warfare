from torch.utils.data import TensorDataset
import torchvision
from torchvision import transforms
import torch.optim as optim
from logging import getLogger
import os
import sys
import argparse
sys.path.append("../")
import utils
from models import Adversary_AE, Watermark_Decoder, OwnerEncoder, StegaStamp
from lpips import lpips
from ssim import *
from utils import EMA
import dataloader
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--csm', '--case_study_model', type=str, default='EDM', choices=['WGAN', 'DDIM', 'EDM'])
parser.add_argument('--dataset', type=str,
					help='which dataset used to train', choices=['cifar10', 'lsun-bedroom', 'celeba', 'ffhq'])
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
					metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--w1_errG', type=float,
					default=10, help='w_G in paper, hyper parameter between adv target & image quality')
parser.add_argument('--w2_errG', type=float,
					default=120, help='w_x in paper, hyper parameter between adv target & image quality')

parser.add_argument('--seed', type=int,
					default=0, help='random seed')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--train_size', default=10000, type=int,
					help='number of adversary samples', choices=[25000, 20000, 15000, 10000, 5000, 2000, 1000])

#EMA settings
parser.add_argument('--model_ema', action='store_true', help='enable EMA of model parameters')
parser.add_argument('--model_ema_steps', type=int, 
					default=16, help='the number of iterations that controls how often to update the EMA')
parser.add_argument('--model_ema_decay', type=float,
					default=0.97,
					help='decay factor for EMA of model parameters')
parser.add_argument('--lr_warmup_epochs', type=int,
					default=50,
					help='keep ema direct copying weights during warmup')

#wandb setting
parser.add_argument('--wandb', action='store_true', help='enable wandb')

#Threat Model settings
parser.add_argument('--target', default='remove', type=str,
					help='threat model type', choices=['remove', 'forge'])
parser.add_argument('--watermark_type', default='text', type=str,
					help='watermark type')
parser.add_argument('--watermark_size', default=64, type=int)
parser.add_argument('--watermark_text', default='0100010001000010111010111111110011101000001111101101010110000000', type=str)
#EDM: 0100010001000010111010111111110011101000001111101101010110000000

parser.add_argument('--dp', action='store_true', help='enable pytorch data parallel')

args = parser.parse_args()

if args.dataset == None:
	if args.csm == 'EDM':
		args.dataset = 'ffhq'
	else:
		args.dataset = 'celeba'

args.save = '../' + args.dataset + '/' + args.dataset + '/' + args.dataset

def calculate_gradient_penalty(model, real_images, fake_images):
	"""Calculates the gradient penalty loss for WGAN GP"""
	# Random weight term for interpolation between real and fake data
	alpha = torch.randn((real_images.size(0), 1, 1, 1)).cuda()
	# Get random interpolation between real and fake data
	interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

	model_interpolates = model(interpolates)
	grad_outputs = torch.ones(model_interpolates.size(), requires_grad=False).cuda()

	# Get gradient w.r.t. interpolates
	gradients = torch.autograd.grad(
		outputs=model_interpolates,
		inputs=interpolates,
		grad_outputs=grad_outputs,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
	return gradient_penalty

if args.dp == False:
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
	os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
	device_ids = [0,1]

utils.setup_seed(args.seed)


logger = getLogger()
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)

print("Watermark size is ", args.watermark_size)
#wd=args.wd
learning_rate=args.lr
w1_errG = args.w1_errG
w2_errG = args.w2_errG
epochs=args.epochs
batch_size=args.batch_size
torch.backends.cudnn.benchmark = True

#load dataset

if args.dataset == 'lsun-bedroom':
	transform_test = transforms.Compose([transforms.Resize((256, 256)),
										transforms.ToTensor(),
										])
	resolution = [256, 256]
elif args.dataset == 'celeba' or args.dataset == 'ffhq':
	transform_test = transforms.Compose([transforms.Resize((64, 64)),
									transforms.ToTensor(),
									])
	resolution = [64, 64]

#write the root for the father folder of the dataset
train_set = dataloader.Paired_Dataset(root='./data/{}/'.format(args.csm), watermark_length=args.watermark_size, transform=transform_test, type=args.target)

dataset_size = len(train_set)
print("dataset_size is", dataset_size)
print("the train size is", args.train_size)
train_indices = random.sample(range(dataset_size), args.train_size)

train_set = torch.utils.data.Subset(train_set, train_indices)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

#load model
if args.dataset == 'cifar10':
	generator = Adversary_AE.ResnetGenerator(3, 3).cuda()
	discriminator = Adversary_AE.Discriminator().cuda()
else:
	generator = Adversary_AE.ResnetGenerator_big(3, 3).cuda()
	discriminator = Adversary_AE.Discriminator_18().cuda()


if args.dp == True:
	generator = torch.nn.DataParallel(generator, device_ids=device_ids)
	discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)
	d_vars = discriminator.module.parameters()
	g_vars = generator.module.parameters()
else:
	d_vars = discriminator.parameters()
	g_vars = generator.parameters()

#load optimizer
if args.dataset == 'cifar10':
	optimize_ae = optim.RMSprop(g_vars, lr=args.lr * 10)
	optimize_dis = optim.RMSprop(d_vars, lr=args.lr * 10)
	INTERVAL = 5
else:
	optimize_ae = optim.Adam(g_vars, lr=0.003, betas=(0.0, 0.99), eps=1e-8)
	optimize_dis = optim.Adam(d_vars, lr=0.003, betas=(0.0, 0.99), eps=1e-8)
	GAN_Loss = utils.LogisticGAN(discriminator)
	INTERVAL = 1

#load loss function
criterion_ae = torch.nn.L1Loss().cuda()
criterion_mse = torch.nn.MSELoss().cuda()
if args.dataset == 'cifar10':
	lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).cuda()
else:
	lpips_loss_fn = lpips.LPIPS(net='vgg', verbose=False).cuda()
global_step = 0

#load EMA model
model_ema_ae = None
if args.model_ema == True:
	adjust = args.batch_size * args.model_ema_steps / args.epochs
	alpha = 1.0 - args.model_ema_decay
	alpha = min(1.0, alpha * adjust)
	print("the dacay is ", 1 - alpha)
	model_ema_ae = EMA(generator, device='cpu', decay = 1 - alpha)

#wnadb
if args.wandb == True:
	import wandb
	wandb.init(
		project="ContentWatermark",
		name='{}_{}_{}_{}'.format(args.csm, args.target, args.w1_errG, args.w2_errG),
		config={
			"Case Study Model": args.csm,
			"target": args.target,
			"watermark_size": args.watermark_size,
			"train_size": args.train_size,
			"w1_errG": args.w1_errG,
			"w2_errG": args.w2_errG,
		}
	)

def test_accuracy():
	watermark = utils.load_watermark(args.watermark_type, ' ',
								 args.watermark_size, args.watermark_text)
	watermark = torch.Tensor(watermark).cuda()

	generator.eval()

	if args.csm == 'EDM':
		ownerencoder = StegaStamp.StegaStampEncoder().cuda()
		ownerencoder.load_state_dict(torch.load('./data/weights/EDM/stegastamp_64_encoder.pth'))
		ownerencoder.eval()

		watermark_decoder = StegaStamp.StegaStampDecoder().cuda()
		watermark_decoder.load_state_dict(torch.load('./data/weights/EDM/stegastamp_64_decoder.pth'))
		watermark_decoder.eval()
	else:
		ownerencoder = OwnerEncoder.Autoencoder(args.watermark_type, args.watermark_size, resolution).cuda()
		ownerencoder.load_state_dict(torch.load(args.save + 'owner_AE_%dbit.pth'%(args.watermark_size)))
		ownerencoder.eval()

		watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()
		watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_%dbit.pth'%(args.watermark_size)))
		watermark_decoder.eval()

	if args.target == 'remove':
		testset = torchvision.datasets.ImageFolder(root='./data/{}/watermark_{}bit_eval'.format(args.csm, args.watermark_size),
											transform=transform_test)
	elif args.target == 'forge':
		data = dataloader.Data(args.dataset, '../data')
		_, testset = data.data_loader(transform_test, transform_test)

	indices = range(10000)
	testset = torch.utils.data.Subset(testset, indices)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
												shuffle=False, drop_last=False, num_workers=0)
	total = 0
	correct = 0

	for images, labels in test_loader:
		total += images.size(0)
		images = images.cuda()
		watermark_ = watermark.tile((images.shape[0], 1)).cuda()
		if args.target == 'remove':
			watermarked = images #use watermark_{}bit_eval dataset
			outputs = generator(watermarked)
		elif args.target == 'forge':
			outputs = generator(images)
		outputs = watermark_decoder(outputs)
		outputs = outputs.detach()
		if args.csm == 'EDM':
			outputs = (outputs > 0).long()
		else:
			outputs = torch.round(outputs)
		correct += (outputs == watermark_).sum().item() / outputs.shape[1]
	accuracy = 100 * correct / total
	return accuracy

if not os.path.exists('./data/weights/{}'.format(args.csm)):
	os.makedirs('./data/weights/{}'.format(args.csm))

#train
for epoch in range(epochs):
	generator.train()
	discriminator.train()
	for i, (images, labels) in enumerate(train_loader):
		global_step += 1
		images = images.cuda().float()
		labels = labels.cuda().float()
		#train generator

		##############################################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		##############################################
		# Set discriminator gradients to zero.
		discriminator.zero_grad()
		optimize_dis.zero_grad()

		# Generate fake image batch with G
		fake_images = generator(images)

		# Train with fake
		if args.dataset != 'cifar10':
			errD = GAN_Loss.dis_loss(labels, fake_images.detach())
		else:
			# Train with real
			real_output = discriminator(labels)
			errD_real = torch.mean(real_output)
			D_x = real_output.mean().item()
			fake_output = discriminator(fake_images.detach())
			errD_fake = torch.mean(fake_output)
			D_G_z1 = fake_output.mean().item()

			# Calculate W-div gradient penalty
			gradient_penalty = calculate_gradient_penalty(discriminator,
														  labels.data, fake_images.data)

			# Add the gradients from the all-real and all-fake batches
			errD = -errD_real + errD_fake + gradient_penalty * 10

		errD.backward()
		# Update D
		if args.dataset == 'cifar10':
			if args.dp == True:
				torch.nn.utils.clip_grad_norm_(discriminator.module.parameters(), 0.01)
			else:
				torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.01)
		optimize_dis.step()

		# Train the generator every n_critic iterations
		if global_step % INTERVAL == 0:
			##############################################
			# (2) Update G network: maximize log(D(G(z)))
			##############################################
			# Set generator gradients to zero
			generator.zero_grad()
			optimize_ae.zero_grad()
			# Generate fake image batch with G
			fake_images = generator(images)

			loss_lpips = torch.mean(lpips_loss_fn(fake_images * 2.0 - 1.0, images * 2.0 - 1.0))
			loss_image = criterion_ae(fake_images, images) + criterion_mse(fake_images, images)

			if args.dataset != 'cifar10':
				errG = w1_errG * GAN_Loss.gen_loss(labels, fake_images) + w2_errG * (loss_lpips + loss_image)
				D_G_z2 = 0.0
			else:
				fake_output = discriminator(fake_images)
				errG = w1_errG * (-torch.mean(fake_output)) + w2_errG * (loss_lpips + loss_image)
				D_G_z2 = fake_output.mean().item()
			errG.backward()
			# Update G
			if args.dataset == 'cifar10':
				if args.dp == True:
					torch.nn.utils.clip_grad_norm_(generator.module.parameters(), 0.25)
				else:
					torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.25)

			optimize_ae.step()
			if model_ema_ae and global_step % args.model_ema_steps == 0:
				model_ema_ae.update_parameters(generator)
				if epoch < args.lr_warmup_epochs:
					model_ema_ae.n_averaged.fill_(0)

		if global_step % 10 == 0:
			print('Epoch [{}/{}], Step [{}/{}], GLoss: {:.4f}, DLoss: {:.4f}'
						.format(epoch + 1, epochs, i + 1, len(train_loader), errG.item(), errD.item()))
			"""
			if args.wandb == True:
				wandb.log(
				{
					'Gloss': errG.item(),
					'Dloss': errD.item(),
				}
			)
			"""
	if (epoch + 1) % 5 == 0:
		print("testing accuracy")
		accuracy = test_accuracy()
		print("accuracy = ", accuracy)
		if args.wandb == True:
			wandb.log({"ep": epoch + 1, "accuracy": accuracy, "image": wandb.Image(torch.cat((images, fake_images), dim=0))})

	if (epoch + 1) % 5 == 0:
		if args.watermark_type == 'text':
			if args.dp == True:
				torch.save(generator.module.state_dict(), './data/weights/%s/'%(args.csm) + '%s_AE_%dbit_%d_%s_%d_w=%d.pth'%(args.csm, args.watermark_size, args.train_size, args.target, epoch+1, args.w2_errG))
				torch.save(discriminator.module.state_dict(), './data/weights/%s/'%(args.csm) + '%s_Discriminator_%dbit_%d_%s_%d_w=%d.pth'%(args.csm, args.watermark_size, args.train_size, args.target, epoch+1, args.w2_errG))
			else:
				torch.save(generator.state_dict(), './data/weights/%s/'%(args.csm) + '%s_AE_%dbit_%d_%s_%d_w=%d.pth'%(args.csm, args.watermark_size, args.train_size, args.target, epoch+1, args.w2_errG))
				torch.save(discriminator.state_dict(), './data/weights/%s/'%(args.csm) + '%s_Discriminator_%dbit_%d_%s_%d_w=%d.pth'%(args.csm, args.watermark_size, args.train_size, args.target, epoch+1, args.w2_errG))
			#save ema model
			if args.model_ema == True:
				torch.save(model_ema_ae.state_dict(),'./data/weights/%s/'%(args.csm) + '%s_AE_%dbit_%d_%s_%d_w=%d_ema.pth'%(args.csm, args.watermark_size, args.train_size, args.target, epoch+1, args.w2_errG))

wandb.finish()

