from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from sim_gan.data_reader import ecg_dataset_pytorch
from tensorboardX import SummaryWriter
from sim_gan.gan_models.models import dcgan
from sim_gan.gan_models.models import vanila_gan
import logging
from sim_gan.data_reader import dataset_configs
import argparse


parser = argparse.ArgumentParser(description='Train an ECG GAN of type Vanilla GAN or DCGAN.', )
parser.add_argument('--GAN_TYPE', type=str, help='Type of gan, either DCGAN or Vanilla GAN.',
                    required=True, choices=['VGAN', 'DCGAN'])
parser.add_argument('--MODEL_DIR', type=str, help='Directory to write summaries and checkpoints.',
                    required=True)
parser.add_argument('--BEAT_TYPE', type=str, help='Type of heartbeat to learn to generate..',
                    required=True,  choices=['N', 'S', 'V', 'F'])
parser.add_argument('--BATCH_SIZE', type=int, help='batch size.',
                    required=True)
parser.add_argument('--NUM_ITERATIONS', type=int, help='Number of iterations.', required=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_ecg_gan(batch_size, num_train_steps, generator, discriminator, model_dir, beat_type):
    # Support for tensorboard:
    writer = SummaryWriter(model_dir)
    # 1. create the ECG dataset:
    # composed = transforms.Compose([ecg_dataset.Scale(), ecg_dataset.Smooth(), ecg_dataset.ToTensor()])

    positive_configs = dataset_configs.DatasetConfigs('train', beat_type, one_vs_all=True, lstm_setting=False,
                                                      over_sample_minority_class=False,
                                                      under_sample_majority_class=False,
                                                      only_take_heartbeat_of_type=beat_type, add_data_from_gan=False,
                                                      gan_configs=None)

    dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(positive_configs,
                                                             transform=ecg_dataset_pytorch.ToTensor())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)
    print("Size of real dataset is {}".format(len(dataset)))

    # 2. Create the models:
    netG = generator
    netD = discriminator

    # Loss functions:
    cross_entropy_loss = nn.BCELoss()

    # Optimizers:
    lr = 0.0002
    beta1 = 0.5
    writer.add_scalar('Learning_Rate', lr)
    optimizer_d = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Noise for validation:
    val_noise = torch.Tensor(np.random.normal(0, 1, (4, 100)))
    loss_d_real_hist = []
    loss_d_fake_hist = []
    loss_g_fake_hist = []
    norma_grad_g = []
    norm_grad_d = []
    d_real_pred_hist = []
    d_fake_pred_hist = []
    epoch = 0
    iters = 0
    while True:
        num_of_beats_seen = 0
        if iters == num_train_steps:
            break
        for i, data in enumerate(dataloader):
            if iters == num_train_steps:
                break

            netD.zero_grad()
            ecg_batch = data['cardiac_cycle'].float()
            b_size = ecg_batch.shape[0]
            num_of_beats_seen += ecg_batch.shape[0]
            output = netD(ecg_batch)
            labels = torch.full((b_size,), 1, device='cpu')
            ce_loss_d_real = cross_entropy_loss(output, labels)
            writer.add_scalar('Discriminator/cross_entropy_on_real_batch', ce_loss_d_real.item(), global_step=iters)
            writer.add_scalars('Merged/losses', {'d_cross_entropy_on_real_batch': ce_loss_d_real.item()},
                               global_step=iters)
            ce_loss_d_real.backward()
            loss_d_real_hist.append(ce_loss_d_real.item())

            mean_d_real_output = output.mean().item()
            d_real_pred_hist.append(mean_d_real_output)

            noise_input = torch.Tensor(np.random.noraml(0, 1, (b_size, 100)))

            output_g_fake = netG(noise_input)
            output = netD(output_g_fake.detach())
            labels.fill_(0)

            ce_loss_d_fake = cross_entropy_loss(output, labels)
            writer.add_scalar('Discriminator/cross_entropy_on_fake_batch', ce_loss_d_fake.item(), iters)
            writer.add_scalars('Merged/losses', {'d_cross_entropy_on_fake_batch': ce_loss_d_fake.item()},
                               global_step=iters)
            ce_loss_d_fake.backward()

            loss_d_fake_hist.append(ce_loss_d_fake.item())

            mean_d_fake_output = output.mean().item()
            d_fake_pred_hist.append(mean_d_fake_output)
            total_loss_d = ce_loss_d_fake + ce_loss_d_real
            writer.add_scalar(tag='Discriminator/total_loss', scalar_value=total_loss_d.item(),
                              global_step=iters)
            optimizer_d.step()

            netG.zero_grad()
            labels.fill_(1)

            output = netD(output_g_fake)
            ce_loss_g_fake = cross_entropy_loss(output, labels)
            ce_loss_g_fake.backward()
            loss_g_fake_hist.append(ce_loss_g_fake.item())
            writer.add_scalar(tag='Generator/cross_entropy_on_fake_batch', scalar_value=ce_loss_g_fake.item(),
                              global_step=iters)
            writer.add_scalars('Merged/losses', {'g_cross_entropy_on_fake_batch': ce_loss_g_fake.item()},
                               global_step=iters)
            mean_d_fake_output_2 = output.mean().item()

            optimizer_g.step()

            print("{}/{}: Epoch #{}: Iteration #{}: Mean D(real_hb_batch) = {}, mean D(G(z)) = {}."
                  .format(num_of_beats_seen, len(dataset), epoch, iters, mean_d_real_output, mean_d_fake_output),
                  end=" ")
            print("mean D(G(z)) = {} After backprop of D".format(mean_d_fake_output_2))

            print("Loss D from real beats = {}. Loss D from Fake beats = {}. Total Loss D = {}".
                  format(ce_loss_d_real, ce_loss_d_fake, total_loss_d), end=" ")
            print("Loss G = {}".format(ce_loss_g_fake))

            # Norma of gradients:
            gNormGrad = get_gradient_norm_l2(netG)
            dNormGrad = get_gradient_norm_l2(netD)
            writer.add_scalar('Generator/gradients_norm', gNormGrad, iters)
            writer.add_scalar('Discriminator/gradients_norm', dNormGrad, iters)
            norm_grad_d.append(dNormGrad)
            norma_grad_g.append(gNormGrad)
            print(
                "Generator Norm of gradients = {}. Discriminator Norm of gradients = {}.".format(gNormGrad, dNormGrad))

            if iters % 25 == 0:
                with torch.no_grad():
                    output_g = netG(val_noise)
                    fig = plt.figure()
                    plt.title("Fake beats from Generator. iteration {}".format(i))
                    for p in range(4):
                        plt.subplot(2, 2, p + 1)
                        plt.plot(output_g[p].detach().numpy(), label="fake beat")
                        plt.plot(ecg_batch[p].detach().numpy(), label="real beat")
                        plt.legend()
                    writer.add_figure('Generator/output_example', fig, iters)
                    plt.close()
            iters += 1
        epoch += 1

    torch.save({
        'epoch': epoch,
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'loss': cross_entropy_loss,

    }, model_dir + '/checkpoint_epoch_{}_iters_{}'.format(epoch, iters))
    writer.close()


def get_gradient_norm_l2(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    model_dir = args.MODEL_DIR
    gan_type = args.GAN_TYPE
    if gan_type == 'VGAN':
        netG = vanila_gan.VGenerator(0)
        netD = vanila_gan.VDiscriminator(0)
    elif gan_type == 'DCGAN':
        netG = dcgan.DCGenerator(0)
        netG.apply(weights_init)

        netD = dcgan.DCDiscriminator(0)
        netD.apply(weights_init)
    else:
        raise ValueError(f"Invalid gan type {gan_type}.")

    train_ecg_gan(args.BATCH_SIZE, args.NUM_ITERATIONS, netG, netD, model_dir, beat_type=args.BEAT_TYPE)
    logging.info("Training completed...")
