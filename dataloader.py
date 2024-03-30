import torchvision
from torchvision.datasets import VisionDataset
import os
from PIL import Image
import torch
class Data:
	def __init__(self, dataset, path):
		self.dataset = dataset
		self.path = path
	def data_loader(self, train_trans, test_trans):
		if self.dataset == 'cifar10':
			trainset = torchvision.datasets.CIFAR10(root=self.path, train=True, download=True, transform=train_trans)
			testset = torchvision.datasets.CIFAR10(root=self.path, train=False, download=True, transform=test_trans)
		elif self.dataset == 'cifar100':
			trainset = torchvision.datasets.CIFAR100(root=self.path, train=True, download=True, transform=train_trans)
			testset = torchvision.datasets.CIFAR100(root=self.path, train=False, download=True, transform=test_trans)
		elif self.dataset == 'svhn':
			trainset = torchvision.datasets.SVHN(root=self.path, split='train', download=True, transform=train_trans)
			testset = torchvision.datasets.SVHN(root=self.path, split='test', download=True, transform=test_trans)
		elif self.dataset == 'imagenet':
			trainset = torchvision.datasets.ImageNet(root=self.path, split='train', transform=train_trans)
			testset = torchvision.datasets.ImageNet(root=self.path, split='val', transform=test_trans)
		elif self.dataset == 'place':
			trainset = torchvision.datasets.Places365(root=self.path, split='train-standard', small=True, download=True, transform=train_trans)
			testset = torchvision.datasets.Places365(root=self.path, split='val', small=True, download=True, transform=test_trans)
		elif self.dataset == 'celeba':
			trainset = torchvision.datasets.CelebA(root=self.path, split='train', download=True, transform=train_trans)
			testset = torchvision.datasets.CelebA(root=self.path, split='valid', download=True, transform=test_trans)
		elif self.dataset == 'lsun-bedroom':
			trainset = torchvision.datasets.LSUN(root = self.path, classes = ['bedroom_train'], transform = train_trans)
			testset = torchvision.datasets.LSUN(root = self.path, classes = ['bedroom_val'], transform = test_trans)
		elif self.dataset == 'ffhq':
			trainset = torchvision.datasets.ImageFolder(root = self.path + '/ffhq', transform = train_trans)
			testset = torchvision.datasets.ImageFolder(root = self.path + '/ffhq', transform = test_trans)
		else:
			ValueError('Unsupported dataset')

		return trainset, testset


class Paired_Dataset(VisionDataset):
	def __init__(self, root, watermark_length, transform=None, target_transform=None, type='remove'):
		super(Paired_Dataset, self).__init__(root, transform=transform, target_transform=target_transform)
		#'watermarked' and 'clean' are the subfolders of root, containing watermarked and clean images respectively
		#the subfolders name can be modified to fit the correct path
		self.watermarked_path = os.path.join(root, 'watermark_%dbit' % (watermark_length))
		self.clean_path = os.path.join(root, 'clean_%dbit' % (watermark_length))
		#self.file_list = sorted(os.listdir(self.clean_path))
		#self.file_list = ['{}.jpeg'.format(i) for i in range (10000)]
		self.file_list = ['{}.pth'.format(i) for i in range(10000)]
		self.type = type

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		watermarked_img_name = os.path.join(self.watermarked_path, self.file_list[idx])
		clean_img_name = os.path.join(self.clean_path, self.file_list[idx])
		#watermarked_img = Image.open(watermarked_img_name)
		watermarked_img = torch.load(watermarked_img_name).detach() #case_study/Adv_AE_train error: requires grad
		#clean_img = Image.open(clean_img_name)
		clean_img = torch.load(clean_img_name).detach()
		#if self.transform:
		#	watermarked_img = self.transform(watermarked_img)
		#	clean_img = self.transform(clean_img)
		if self.type == 'remove':
			return watermarked_img, clean_img
		elif self.type == 'forge':
			return clean_img, watermarked_img
		else:
			ValueError('Unsupported type')

class Paired_Dataset_fewshot(VisionDataset):
	def __init__(self, root, watermark_length, transform=None, target_transform=None, type='remove'):
		super(Paired_Dataset_fewshot, self).__init__(root, transform=transform, target_transform=target_transform)
		#'watermarked' and 'clean' are the subfolders of root, containing watermarked and clean images respectively
		#the subfolders name can be modified to fit the correct path
		self.watermarked_path = os.path.join(root, 'fewshot_watermark_%dbit' % (watermark_length))
		self.clean_path = os.path.join(root, 'fewshot_clean_%dbit' % (watermark_length))
		#self.file_list = sorted(os.listdir(self.clean_path))
		#self.file_list = ['{}.jpeg'.format(i) for i in range (10000)]
		self.file_list = ['{}.pth'.format(i) for i in range(100)]
		self.type = type

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		watermarked_img_name = os.path.join(self.watermarked_path, self.file_list[idx])
		clean_img_name = os.path.join(self.clean_path, self.file_list[idx])
		#watermarked_img = Image.open(watermarked_img_name)
		watermarked_img = torch.load(watermarked_img_name)
		#clean_img = Image.open(clean_img_name)
		clean_img = torch.load(clean_img_name)
		#if self.transform:
		#	watermarked_img = self.transform(watermarked_img)
		#	clean_img = self.transform(clean_img)
		if self.type == 'remove':
			return watermarked_img, clean_img
		elif self.type == 'forge':
			return clean_img, watermarked_img
		else:
			ValueError('Unsupported type')

