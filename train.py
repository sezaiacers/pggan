import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import math

import argparse
import pggan

import shutil
import os


def build_loader(data_dir, resolution, input_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    loader = iter(dataloader)
    while True:
        try:
            images, _ = next(loader)
        except StopIteration:
            loader = iter(dataloader)
            image, _ = next(loader)
        yield images

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4, 1024: 4}

    global_step, cur_nimg, training_nimg, transition_nimg = 0, 0, args.training_kimg * 1000, args.transition_kimg * 1000

    generator = pggan.build_generator(args.latent_size, 0, args.resolution).to(device=device)
    discriminator = pggan.build_discriminator(args.resolution, args.num_channels, 0, mbstd_group_size=args.mbstd_group_size).to(device=device)


    generator_optimizer = optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(0.0, 0.99))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(0.0, 0.99))

    resume = args.resume and os.path.isfile(args.resume)
    if resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)

        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])

        generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

        cur_nimg = checkpoint['cur_nimg']
        global_step = checkpoint['global_step']
    else:
        shutil.rmtree(args.images, ignore_errors=True)
        shutil.rmtree(args.checkpoint, ignore_errors=True)
        os.makedirs(args.images, exist_ok=True)
        os.makedirs(args.checkpoint, exist_ok=True)


    def save(cur_nimg, global_step):
        checkpoint = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'cur_nimg': cur_nimg,
            'global_step': global_step,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint, '{}.pth.tar'.format(cur_nimg)))


    generator.train()
    discriminator.train()

    # print
    n, alpha = -1, 0.0
    while True:
        new_n = cur_nimg // (training_nimg + transition_nimg)
        if n != new_n:
            if n != -1:
                generator_optimizer = optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(0.0, 0.99))
                discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(0.0, 0.99))
                save(cur_nimg, global_step)
            n = new_n
            input_size = 2 ** (n + 2)
            batch_size = batch_size_dict[input_size]
            loader = build_loader(args.data_dir, args.resolution, input_size, batch_size, args.num_workers)

        alpha = (cur_nimg - (training_nimg + transition_nimg) * n) / transition_nimg
        for _ in range(args.num_critic):
            latents = torch.randn((batch_size, args.latent_size), device=device)
            fakes = generator(latents, None, n, alpha).detach()
            reals = next(loader).to(device=device)

            fake_scores = discriminator(fakes, n, alpha)
            real_scores = discriminator(reals, n, alpha)

            mixing_factors = torch.rand((batch_size, 1, 1, 1), device=device)
            mixed_images = reals * mixing_factors + fakes * (1.0 - mixing_factors)
            mixed_images.requires_grad = True
            mixed_scores = discriminator(mixed_images, n, alpha)

            grads = autograd.grad(outputs=mixed_scores, inputs=mixed_images, \
                                  grad_outputs=torch.ones_like(mixed_scores), \
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

            grads = grads.reshape(grads.size(0), -1)
            gradient_penalty = ((grads.norm(2, dim=1) - 1) ** 2)

            discriminator_loss = fake_scores.mean() - real_scores.mean() + gradient_penalty.mean() * args.wgan_lambda

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            print('Global Step: {}, N: {}, Alpha: {:.2f}, Discriminator Loss: {:.2f}'.format(global_step, n, alpha, discriminator_loss.item()))

        latents = torch.randn((batch_size, args.latent_size), device=device)
        fakes = generator(latents, None, n, alpha)
        generator_loss = -discriminator(fakes, n, alpha).mean()

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        print('Global Step: {}, N: {}, Alpha: {:.2f}, Generator Loss: {:.2f}'.format(global_step, n, alpha, generator_loss.item()))

        if global_step % args.save_freq == 0:
            save(cur_nimg, global_step)

        cur_nimg += batch_size
        global_step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=40, type=int, metavar='N')

    parser.add_argument('--data-dir', type=str, metavar='DIR', required=True)
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N')

    parser.add_argument('--resolution', type=int, default=32, metavar='N')
    parser.add_argument('--latent-size', type=int, default=512, metavar='N')
    parser.add_argument('--num-channels', type=int, default=3, metavar='N')
    parser.add_argument('--mbstd-group-size', type=int, default=4, metavar='N')


    parser.add_argument('--generator-lr', type=float, default=0.001, metavar='F')
    parser.add_argument('--discriminator-lr', type=float, default=0.001, metavar='F')
    parser.add_argument('--wgan-lambda', type=float, default=10.0, metavar='F')

    parser.add_argument('--num-critic', type=int, default=4, metavar='N')

    parser.add_argument('--training-kimg' , type=int, default=600, metavar='N')
    parser.add_argument('--transition-kimg', type=int, default=600, metavar='N')

    parser.add_argument('--save_freq', type=int, default=10000, metavar='N')

    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--images', type=str, default='images', metavar='DIR')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', metavar='DIR')

    main(parser.parse_args())
