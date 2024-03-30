import torch
from torch import nn
from models.utils import Conv2D, Dense, Flatten
from torch.nn import functional as F

class Autoencoder(nn.Module):#to extract embeddings (if the secret message is an image)
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.conv1 = Conv2D(3, 32, 3, activation='relu')
		self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
		self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
		self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
		self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
		self.up6 = Conv2D(256, 128, 3, activation='relu')
		self.conv6 = Conv2D(256, 128, 3, activation='relu')
		self.up7 = Conv2D(128, 64, 3, activation='relu')
		self.conv7 = Conv2D(128, 64, 3, activation='relu')
		self.up8 = Conv2D(64, 32, 3, activation='relu')
		self.conv8 = Conv2D(64, 32, 3, activation='relu')
		self.up9 = Conv2D(32, 32, 3, activation='relu')
		self.conv9 = Conv2D(67, 32, 3, activation='relu')
		self.residual = Conv2D(32, 3, 1, activation=None)

	def forward(self, image):
		image = image - .5
		conv1 = self.conv1(image) #32
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
		merge6 = torch.cat([conv4, up6], dim=1)
		conv6 = self.conv6(merge6)
		up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
		merge7 = torch.cat([conv3, up7], dim=1)
		conv7 = self.conv7(merge7)
		up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
		merge8 = torch.cat([conv2, up8], dim=1)
		conv8 = self.conv8(merge8)
		up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8)) #32
		merge9 = torch.cat([conv1, up9, image], dim=1) #32+32+3 = 67
		conv9 = self.conv9(merge9)
		residual = self.residual(conv9)
		residual = (F.tanh(residual) + 1.) / 2.
		return residual

class SpatialTransformerNetwork(nn.Module):
	def __init__(self, dataset):
		super(SpatialTransformerNetwork, self).__init__()
		self.localization = nn.Sequential(
			Conv2D(3, 32, 3, strides=2, activation='relu'),
			Conv2D(32, 64, 3, strides=2, activation='relu'),
			Conv2D(64, 128, 3, strides=2, activation='relu'),
			nn.AdaptiveAvgPool2d(4),
			Flatten(),
			Dense(2048, 128, activation='relu'),
			nn.Linear(128, 6)
		)
		self.localization[-1].weight.data.zero_()
		self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

	def forward(self, image):
		theta = self.localization(image)
		theta = theta.view(-1, 2, 3)
		grid = F.affine_grid(theta, image.size())
		transformed_image = F.grid_sample(image, grid)
		return transformed_image

class Decoder(nn.Module):
	def __init__(self, watermark_type, watermark_size, dataset):
		super(Decoder, self).__init__()
		self.watermark_size = watermark_size
		self.watermark_type = watermark_type

		self.stn = SpatialTransformerNetwork(dataset)  # develop robustness against small perspective changes
												# that are introduced while capturing and rectifying
												# the encoded image
		if self.watermark_type == 'text':
			self.decoder = nn.Sequential(
				Conv2D(3, 32, 3, strides=2, activation='relu'),
				Conv2D(32, 32, 3, activation='relu'),
				Conv2D(32, 64, 3, strides=2, activation='relu'),
				Conv2D(64, 64, 3, activation='relu'),
				Conv2D(64, 64, 3, strides=2, activation='relu'),
				Conv2D(64, 128, 3, strides=2, activation='relu'),
				Conv2D(128, 128, 3, strides=2, activation='relu'),
				nn.AdaptiveAvgPool2d(1),
				Flatten(),
				Dense(128, 64, activation='relu'),
				Dense(64, self.watermark_size, activation=None),
				nn.Sigmoid())
		else:
			self.decoder = Autoencoder()

	def forward(self, image):
		#print("---------in Decoder--------")
		image = image - .5
		#print("image size is", image.size())
		transformed_image = self.stn(image)
		#print("transformed size is", transformed_image.size())
		decodered = self.decoder(transformed_image)
		#print("decoded size is", decodered.size())
		return decodered