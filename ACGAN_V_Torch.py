import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)


class opt:
    n_epochs = 5000
    batch_size = 32
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 8
    latent_dim = 150
    n_classes = 10  # Set this to 10 for CIFAR-10
    img_size = 32
    channels = 3
    sample_interval = 2000


print(opt)

cuda = True if torch.cuda.is_available() else False


# def weights_init_normal(m):
#   classname = m.__class__.__name__
#  if classname.find("Conv") != -1:
#     torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
# elif classname.find("BatchNorm2d") != -1:
#   torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#  torch.nn.init.constant_(m.bias.data, 0.0)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def load_cifar10():
    dataroot = 'C:\\Users\\user\\Desktop\\jinmo\\cifar10\\cifar-10-batches-py'  # Adjust this path if needed
    dataset = datasets.CIFAR10(root=dataroot, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((opt.img_size, opt.img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ]))

    # Filter out all examples except for cars
    indices = np.where(np.array(dataset.targets) == 1)[0]
    dataset.data = dataset.data[indices]
    dataset.targets = np.array(dataset.targets)[indices].tolist()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return dataloader


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 256 * self.init_size ** 2))  # Increase the number of filters

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),  # Increase the number of filters
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # Increase the number of filters
            nn.BatchNorm2d(256, 0.8),  # Increase the number of filters
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),  # Increase the number of filters
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(),
            nn.Conv2d(128, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 32, bn=False),  # Increase the number of filters
            *discriminator_block(32, 64),  # Increase the number of filters
            *discriminator_block(64, 128),  # Increase the number of filters
            *discriminator_block(128, 256),  # Increase the number of filters
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())  # Increase the number of filters
        self.aux_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, opt.n_classes),
                                       nn.Softmax())  # Increase the number of filters

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = load_cifar10()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = Variable(LongTensor([i // n_row for i in range(n_row ** 2)]))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images1/%d.png" % batches_done, nrow=n_row, normalize=True)


# Training loop
D_losses, G_losses, D_accuracies = [], [], []
# Training loop
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.full((batch_size,), 3)))  # always generate cats

        # gen_labels = Variable(LongTensor(np.full((batch_size,), 1)))  # always generate cars

        gen_labels = Variable(LongTensor(np.random.randint(1, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, gen_labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([gen_labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())
        D_accuracies.append(100 * d_acc)

        d_loss.backward()
        optimizer_D.step()

        # Print training information
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %.2f%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )
      # Here you compute the average losses and accuracy for each epoch
    avg_D_loss = np.mean(D_losses)
    avg_G_loss = np.mean(G_losses)
    avg_D_acc = np.mean(D_accuracies)

    print(
        "[Epoch %d/%d] [Avg D loss: %f, avg acc: %.2f%%] [Avg G loss: %f]"
        % (epoch, opt.n_epochs, avg_D_loss, avg_D_acc, avg_G_loss)
    )