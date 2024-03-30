import argparse
import torch
import torchvision.utils as vutils
import os
import sys
sys.path.append("./WassersteinGAN_DIV-PyTorch-master/")
import wgandiv_pytorch.models as w_model
from wgandiv_pytorch import create_folder
from wgandiv_pytorch import select_device
sys.path.append("../")
from models import Watermark_Decoder
import utils

parser = argparse.ArgumentParser(description="An implementation of WassersteinGAN-DIV algorithm using PyTorch framework.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="lsun",
                    help="model architecture:  (default: lsun)")
parser.add_argument("-n", "--num-images", type=int, default=64,
                    help="How many samples are generated at one time. (default: 64).")
parser.add_argument("--outf", default="test", type=str, metavar="PATH",
                    help="The location of the image in the evaluation process. (default: ``test``).")
parser.add_argument('--gpu', type=int, default=3, help='GPU to use [default: GPU 0]')
parser.add_argument('--weight_number', type=int, default=100000, help='choose the weights to run')

parser.add_argument('--dataset', type=str, default='celeba')

#Watermark settings
parser.add_argument('--watermark_type', default='text', type=str,
                    help='watermark type')
parser.add_argument('--watermark_size', default=32, type=int)
parser.add_argument('--watermark_path', default='watermark.png', type=str)
parser.add_argument('--watermark_text', default='10001000100010001000100010001000', type=str)

parser.add_argument('--size', default=10000, type=int, help='number of images for the model to generate')

parser.add_argument('--eval_gen', action='store_true', help='enable generating images for evaluation')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
weight_path = 'WassersteinGAN_DIV-PyTorch-master/weights/lsun_G_iter_{}.pth'.format(args.weight_number)

def test_accuracy(images):
    watermark = utils.load_watermark(args.watermark_type, ' ',
                                    args.watermark_size, args.watermark_text)
    watermark = torch.Tensor(watermark).cuda()

    watermark_decoder = Watermark_Decoder.Decoder(args.watermark_type, args.watermark_size, args.dataset).cuda()
    watermark_decoder.load_state_dict(torch.load('../' + args.dataset + '/' + args.dataset + '/' + args.dataset + 'owner_Decoder_%dbit.pth'%(args.watermark_size)))
    watermark_decoder.eval()


    total = 0
    correct = 0

    total += images.size(0)
    images = images.cuda()
    watermark_ = watermark.tile((images.shape[0], 1)).cuda()
    outputs = watermark_decoder(images)
    outputs = outputs.detach()
    outputs = torch.round(outputs)
    correct += (outputs == watermark_).sum().item() / outputs.shape[1]
    accuracy = 100 * correct / total
    return accuracy

#load model
print("loading model")
model = w_model.__dict__[args.arch]().cuda()
#model = torch.hub.load("Lornatang/WassersteinGAN_DIV-PyTorch", args.arch, progress=True, pretrained=False, verbose=False).cuda()
model.load_state_dict(torch.load(weight_path))

model.eval()

if args.eval_gen == True:
    save_path = '/data/WGAN/watermark_{}bit_eval/to'.format(args.watermark_size)
else:
    save_path = '/data/WGAN/watermark_{}bit'.format(args.watermark_size)

cwd = os.getcwd()
print('cwd is: ', cwd)
if not os.path.exists(cwd + save_path):
    os.makedirs(cwd + save_path)

def generate():
    count = 0
    while(count < args.size):
        noise = torch.randn(args.num_images, 100, 1, 1, device='cuda')
        generated_images = model(noise)
        #print(generated_images.size())

        for j in range(generated_images.size(0)):
            pic_id = count + j
            if pic_id > args.size:
                break
            #vutils.save_image(generated_images[j].cpu(), cwd + '/data/WGAN/watermark_{}bit/{}.png'.format(args.watermark_size, pic_id))
            if args.eval_gen == True:
                vutils.save_image(generated_images[j].cpu(), cwd + save_path + '/{}.png'.format(pic_id))
            else:
                torch.save(generated_images[j].cpu(), cwd + save_path + '/{}.pth'.format(pic_id))
        count += generated_images.size(0)
        
        if count % 5 == 0:
            accuracy = test_accuracy(generated_images)
            print("accuracy is: ", accuracy)
            vutils.save_image(generated_images, cwd + '/data/WGAN/test.png', normalize=True)
            print('{}/{}'.format(count, args.size))


if __name__ == '__main__':
    generate()