import torch
from torch import nn
from models.utils import Conv2D, Dense
# refer this paper: https://github.com/tancik/StegaStamp

class Autoencoder(nn.Module):
	def __init__(self, watermark_type, watermark_size, resolution:list):
		super(Autoencoder, self).__init__()
		self.watermark_type = watermark_type
		self.watermark_size = watermark_size
		self.resolution = resolution
		if self.watermark_type == 'text':
			if resolution[0] > 50:
				self.secret_dense = nn.Sequential(Dense(self.watermark_size, 32*32*3,
				                          activation='relu', kernel_initializer='he_normal'),
				                                  nn.Unflatten(1 , (3, 32, 32)),
				                                  nn.Upsample(size=(resolution[0], resolution[1])))

			else:
				self.secret_dense = Dense(self.watermark_size, self.resolution[0]*self.resolution[1]*3,
			                          activation='relu', kernel_initializer='he_normal')

		self.conv1 = Conv2D(6, 32, 3, activation='relu')
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
		self.conv9 = Conv2D(70, 32, 3, activation='relu')
		self.residual = Conv2D(32, 3, 1, activation=None)

	def forward(self, image, message):
		message = message - .5
		image = image - .5
		if self.watermark_type == 'text':
			message = self.secret_dense(message)
			message = message.reshape(-1, 3, self.resolution[0], self.resolution[1])
		inputs = torch.cat([image, message], dim=1)
		conv1 = self.conv1(inputs)
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
		up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
		merge9 = torch.cat([conv1, up9, inputs], dim=1)
		conv9 = self.conv9(merge9)
		residual = self.residual(conv9)
		return residual

