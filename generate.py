import torch
import pggan
import math
import argparse
import os

from torchvision.utils import save_image


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = pggan.build_generator(args.latent_size, 0, args.resolution).to(device=device)

    print("=> loading checkpoint '{}'".format(args.resume))
    generator.load_state_dict(torch.load(args.resume, map_location=device)['generator'])

    generator.eval()
    latents = torch.randn((args.batch_size, args.latent_size), device=device) * args.temperature

    images = generator(latents, None, args.n, 0.1).cpu()

    os.makedirs(args.images, exist_ok=True)
    save_image(images, os.path.join(args.images, args.name), nrow=int(math.sqrt(args.batch_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=40, type=int, metavar='N')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')

    parser.add_argument('--resolution', type=int, default=128, metavar='N')
    parser.add_argument('--latent-size', type=int, default=512, metavar='N')
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--images', type=str, default='images', metavar='DIR')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--n', type=int, required=True, metavar='N')

    parser.add_argument('--temperature', type=float, default=1.0, metavar='F')

    main(parser.parse_args())
