import torch
import torch.utils.data
import logging
import random
import numpy as np
import time
from datetime import timedelta
import copy
import os
import random
from torch import nn

CIFAR10_classes = ["airplane", "automobile", "bird",  "cat",  "deer",
				   "dog",      "frog",       "horse", "ship", "truck"]
CIFAR10_classes = ["0", "1", "2",  "3",  "4",
				   "5",      "6",       "7", "8", "9"]
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))

def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.

class LogFormatter:
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime("%x %X"),
			timedelta(seconds=elapsed_seconds),
		)
		message = record.getMessage()
		message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
		return "%s - %s" % (prefix, message) if message else ""

def create_logger(filepath, rank):
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	if filepath is not None:
		if rank > 0:
			filepath = "%s-%i" % (filepath, rank)
		file_handler = logging.FileHandler(filepath, "a")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	if filepath is not None:
		logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)

def load_watermark(type, path, length, value):
	if type == "image":
		if not os.path.exists(path):
			raise ValueError("The watermark image does not exist")
		else:
			# load image from path
			# resize image to 32x32
			pass
	elif type == "text":
		bits = []
		if len(value) == length:
			for i in range(length):
				if value[i] != "0" and value[i] != "1":
					raise ValueError("The inputted watermark is not binary")
				else:
					bits.append(int(value[i]))
			return bits
		else:
			raise ValueError("The length of inputted watermark is not equal to the target length")
def split_dataset(dataset, total_number, save_path, train_ids=None, val_ids=None):
	if (train_ids is None) and (val_ids is None):
		target_list = np.asarray(dataset.targets).tolist()
		total_data = len(dataset.data)
		num_classes = len(list(set(target_list)))
		print(num_classes)
		new_size = total_number

		num_pre_class_new = new_size // num_classes

		split_idx = {}
		split_val_ids = {}
		dict_new = {}

		for i in range(num_classes):
			dict_new[i] = 0
			split_idx[i] = []
			split_val_ids[i] = []

		random_idx = [i for i in range(total_data)]
		for i in range(num_classes):
			for idx in random_idx:
				if target_list[idx] == i:
					if dict_new[i] < num_pre_class_new:
						split_idx[i].append(idx)
						dict_new[i] += 1
					else:
						split_val_ids[i].append(idx)
		torch.save(split_idx, os.path.join(save_path, "train_idx.pth"))
		torch.save(split_val_ids, os.path.join(save_path, "val_idx.pth"))
		data_list = []
		label_list = []
		for i in range(num_classes):
			data_list.append(dataset.data[split_idx[i]])
			label_list.append(np.asarray(dataset.targets)[split_idx[i]])
		new_dataset = copy.deepcopy(dataset)
		new_dataset.data = np.vstack(data_list)
		new_dataset.targets = np.vstack(label_list).reshape(new_dataset.data.shape[0],).tolist()
	else:
		target_list = np.asarray(dataset.targets).tolist()
		num_classes = len(list(set(target_list)))
		print(num_classes)
		data_list = []
		label_list = []
		for i in range(num_classes):
			data_list.append(dataset.data[train_ids[i]])
			label_list.append(np.asarray(dataset.targets)[train_ids[i]])
		new_dataset = copy.deepcopy(dataset)
		new_dataset.data = np.vstack(data_list)
		new_dataset.targets = np.vstack(label_list).reshape(new_dataset.data.shape[0],).tolist()
	return new_dataset

def split_dataset_lsun(dataset, total_number, save_path, train_ids=None, val_ids=None):
	dataset_size = len(dataset)
	random.seed(0)
	train_indices = random.sample(range(dataset_size), total_number)
	#test_indices = [i for i in range(dataset_size) if i not in train_indices]
	new_dataset = torch.utils.data.Subset(dataset, train_indices)
	#testset = torch.utils.data.Subset(dataset, test_indices)
	testset= None
	return new_dataset, testset

def split_dataset_celeba(dataset, total_number, save_path, train_ids=None, val_ids=None):
	dataset_size = len(dataset)
	random.seed(0)
	train_indices = random.sample(range(dataset_size), total_number)
	#test_indices = [i for i in range(dataset_size) if i not in train_indices]
	new_dataset = torch.utils.data.Subset(dataset, train_indices)
	#testset = torch.utils.data.Subset(dataset, test_indices)
	testset = None
	return new_dataset, testset

import math
import cv2
def psnr(x1, x2):
	mse = np.mean( (x1 - x2) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim_(x1, x2):
	C1 = (0.01 * 255) ** 2
	C2 = (0.03 * 255) ** 2
	img1 = x1.astype(np.float64)
	img2 = x2.astype(np.float64)
	kernel = cv2.getGaussianKernel(11, 1.5)
	window = np.outer(kernel, kernel.transpose())
	mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
	mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
	mu1_sq = mu1 ** 2
	mu2_sq = mu2 ** 2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
	sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
	sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
															(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()

from torch.optim.swa_utils import AveragedModel

class EMA(AveragedModel):
	def __init__(self, model, decay, device):
		def ema_avg(avg_model_param, model_param, num_averaged): #define the ema average function and pass to the AveragedModel
			return decay * avg_model_param + (1 - decay) * model_param
		
		super().__init__(model, device, ema_avg, use_buffers=True)


class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")

from torch import nn
class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img):

        # TODO: use_loss_scaling, for fp16
        #apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        #undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach()) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps):
        f_preds = self.dis(fake_samps)

        return torch.mean(nn.Softplus()(-f_preds))