import torch
import argparse
import sys
import os
import random
from torchvision.utils import save_image
from torchvision import transforms
origin_sys_path = sys.path[:]
sys.path.append("..")
import utils
import dataloader
from models import OwnerEncoder, Watermark_Decoder
sys.path = origin_sys_path

parser = argparse.ArgumentParser()

parser.add_argument('--gpus', type=str, default='3')
parser.add_argument('--dataset', default='celeba', type=str,
					help='which dataset used to train', choices=['cifar10', 'lsun-bedroom', 'celeba'])
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--watermark_type', default='text', type=str,
                    help='watermark type')
parser.add_argument('--watermark_size', default=32, type=int)
parser.add_argument('--watermark_path', default='watermark.png', type=str)
parser.add_argument('--watermark_text', default='10001000100010001000100010001000', type=str)

parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--size', default=50000, type=int, help='number of watermark images to generate')

args = parser.parse_args()
args.save = '../' + args.dataset + '/' + args.dataset + '/' + args.dataset
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

torch.manual_seed(args.seed)
random.seed(args.seed)

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

data = dataloader.Data(args.dataset, '../data')
trainset, testset = data.data_loader(transform_test, transform_test)
trainset = torch.utils.data.Subset(trainset, range(args.size))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, drop_last=False, num_workers=0)


autoencoder = OwnerEncoder.Autoencoder(args.watermark_type, args.watermark_size, resolution=resolution).cuda()
watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()

if args.watermark_type == 'text':
	autoencoder.load_state_dict(torch.load(args.save + 'owner_AE_%dbit.pth'%(args.watermark_size)))
	watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_%dbit.pth'%(args.watermark_size)))
elif args.watermark_type == 'image':
	autoencoder.load_state_dict(torch.load(args.save + 'owner_AE_img.pth'))
	watermark_decoder.load_state_dict(torch.load(args.save + 'owner_Decoder_img.pth'))

autoencoder.eval()
watermark_decoder.eval()

watermark = utils.load_watermark(args.watermark_type, args.watermark_path,
                                args.watermark_size, args.watermark_text)
watermark = torch.Tensor(watermark).cuda()

cwd = os.getcwd()
print(os.getcwd())
def generate_watermark():
	if not os.path.exists(cwd + '/data/{}_watermarked_{}bit/to'.format(args.dataset, args.watermark_size)):
		os.makedirs(cwd + '/data/{}_watermarked_{}bit/to'.format(args.dataset, args.watermark_size))

	for i, (images, labels) in enumerate(train_loader):
		images = images.cuda()

		if args.watermark_type == 'text':
			watermark_ = watermark.tile((images.shape[0], 1)).cuda()
		else:
			watermark_ = watermark.tile((images.shape[0], 1, 1, 1)).cuda()

		wm_images = autoencoder(images, watermark_) + images #residual is needed!
		wm_images = wm_images.detach()
		#print(images)
		print(wm_images)

		for j in range(wm_images.size(0)):
			pic_id = j + i * args.batch_size
			save_image(wm_images[j], cwd + '/data/{}_watermarked_{}bit/to/{}.png'.format(args.dataset, args.watermark_size, pic_id))
		print('{}/{}'.format(i, len(train_loader)))

if __name__ == '__main__':
    generate_watermark()