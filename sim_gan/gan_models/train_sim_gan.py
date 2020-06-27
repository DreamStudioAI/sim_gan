import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sim_gan.data_reader import ecg_dataset_pytorch
from tensorboardX import SummaryWriter
from sim_gan.gan_models.models import sim_gan_euler
from sim_gan.dynamical_model import equations
from sim_gan.dynamical_model.ode_params import ODEParams
import math
import logging
from sim_gan.dynamical_model import typical_beat_params
from sim_gan.gan_models.models import vanila_gan
from sim_gan.data_reader import dataset_configs
import argparse


parser = argparse.ArgumentParser(description='Train an SIM ECG GAN of type Vanilla GAN or DCGAN.', )
parser.add_argument('--GAN_TYPE', type=str, help='Type of gan, either SimDCGAN or SimVGAN.',
                    required=True, choices=['SimVGAN', 'SimDCGAN'])
parser.add_argument('--MODEL_DIR', type=str, help='Directory to write summaries and checkpoints.',
                    required=True)
parser.add_argument('--BEAT_TYPE', type=str, help='Type of heartbeat to learn to generate..',
                    required=True,  choices=['N', 'S', 'V', 'F'])
parser.add_argument('--BATCH_SIZE', type=int, help='batch size.',
                    required=True)
parser.add_argument('--NUM_ITERATIONS', type=int, help='Number of iterations.', required=True)


TYPICAL_ODE_N_PARAMS = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                      160.0 * math.pi / 180.0]


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_typical_N_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(TYPICAL_ODE_N_PARAMS).to(device)
    return params


def generate_typical_S_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_S_PARAMS).to(device)
    return params


def generate_typical_F_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_F_PARAMS).to(device)
    return params

def generate_typical_V_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_V_PARAMS).to(device)
    return params


def ode_loss(hb_batch, ode_params, device, beat_type):
    """

    :param hb_batch:
    :return:
    """
    delta_t = ode_params.h
    batch_size = hb_batch.size()[0]
    if beat_type == "N":
        params_batch = generate_typical_N_ode_params(batch_size, device)
    elif beat_type == "S":
        params_batch = generate_typical_S_ode_params(batch_size, device)
    elif beat_type == 'F':
        params_batch = generate_typical_F_ode_params(batch_size, device)
    elif beat_type == 'V':
        params_batch = generate_typical_V_ode_params(batch_size, device)
    else:
        raise NotImplementedError()

    logging.debug("params batch shape: {}".format(params_batch.size()))
    x_t = torch.tensor(-0.417750770388669).to(device)
    y_t = torch.tensor(-0.9085616622823985).to(device)
    t = torch.tensor(0.0).to(device)
    f_ode_z_signal = None
    delta_hb_signal = None
    for i in range(215):
        delta_hb = (hb_batch[:, i + 1] - hb_batch[:, i]) / delta_t
        delta_hb = delta_hb.view(-1, 1)
        z_t = hb_batch[:, i].view(-1, 1)

        f_ode_x = equations.d_x_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_y = equations.d_y_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_z = equations.d_z_d_t(x_t, y_t, z_t, t, params_batch, ode_params)

        logging.debug("f ode z shape {}".format(f_ode_z.shape))  # Nx1
        logging.debug("f ode x shape {}".format(f_ode_x.shape))
        logging.debug("f ode y shape {}".format(f_ode_y.shape))

        y_t = y_t + delta_t * f_ode_y
        x_t = x_t + delta_t * f_ode_x
        t += 1 / 360

        if f_ode_z_signal is None:
            f_ode_z_signal = f_ode_z
            delta_hb_signal = delta_hb
        else:
            f_ode_z_signal = torch.cat((f_ode_z_signal, f_ode_z), 1)
            delta_hb_signal = torch.cat((delta_hb_signal, delta_hb), 1)
    logging.debug("f signal shape: {}".format(f_ode_z_signal.shape))
    logging.debug("delta hb signal shape: {}".format(delta_hb_signal.shape))
    return delta_hb_signal, f_ode_z_signal


def euler_loss(hb_batch, params_batch, x_batch, y_batch, ode_params):
    """

    :param hb_batch: Nx216
    :param params_batch: Nx15
    :return:
    """
    logging.debug('hb batch shape: {}'.format(hb_batch.shape))
    logging.debug('params batch shape: {}'.format(params_batch.shape))
    logging.debug('x batch shape: {}'.format(x_batch.shape))
    logging.debug('y batch shape: {}'.format(y_batch.shape))

    delta_t = ode_params.h
    t = torch.tensor(0.0)
    f_ode_z_signal = None
    f_ode_x_signal = None
    f_ode_y_signal = None
    delta_hb_signal = None
    delta_x_signal = None
    delta_y_signal = None
    for i in range(215):
        delta_hb = (hb_batch[:, i + 1] - hb_batch[:, i]) / delta_t
        delta_y = (y_batch[:, i + 1] - y_batch[:, i]) / delta_t
        delta_x = (x_batch[:, i + 1] - x_batch[:, i]) / delta_t
        delta_hb = delta_hb.view(-1, 1)
        delta_x = delta_x.view(-1, 1)
        delta_y = delta_y.view(-1, 1)
        logging.debug("Delta heart-beat shape: {}".format(delta_hb.shape))
        y_t = y_batch[:, i].view(-1, 1)
        x_t = x_batch[:, i].view(-1, 1)
        z_t = hb_batch[:, i].view(-1, 1)
        f_ode_x = equations.d_x_d_t(y_t, x_t, t,  ode_params.rrpc, ode_params.h)
        f_ode_y = equations.d_y_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_z = equations.d_z_d_t(x_t, y_t, z_t, t, params_batch, ode_params)
        t += 1 / 512

        logging.debug("f ode z shape {}".format(f_ode_z.shape))  # Nx1
        logging.debug("f ode x shape {}".format(f_ode_x.shape))
        logging.debug("f ode y shape {}".format(f_ode_y.shape))
        if f_ode_z_signal is None:
            f_ode_z_signal = f_ode_z
            f_ode_x_signal = f_ode_x
            f_ode_y_signal = f_ode_y
            delta_hb_signal = delta_hb
            delta_x_signal = delta_x
            delta_y_signal = delta_y
        else:
            f_ode_z_signal = torch.cat((f_ode_z_signal, f_ode_z), 1)
            f_ode_x_signal = torch.cat((f_ode_x_signal, f_ode_x), 1)
            f_ode_y_signal = torch.cat((f_ode_y_signal, f_ode_y), 1)
            delta_hb_signal = torch.cat((delta_hb_signal, delta_hb), 1)
            delta_x_signal = torch.cat((delta_x_signal, delta_x), 1)
            delta_y_signal = torch.cat((delta_y_signal, delta_y), 1)


    logging.debug("f signal shape: {}".format(f_ode_z_signal.shape))
    logging.debug("delta hb signal shape: {}".format(delta_hb_signal.shape))

    return delta_hb_signal, f_ode_z_signal, f_ode_x_signal, f_ode_y_signal, delta_x_signal, delta_y_signal


def train(batch_size, num_train_steps, model_dir, beat_type, generator_net, discriminator_net):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ode_params = ODEParams(device)

    #
    # Support for tensorboard:
    #
    writer = SummaryWriter(model_dir)

    #
    # 1. create the ECG dataset:
    #
    composed = transforms.Compose([ecg_dataset_pytorch.Scale(), ecg_dataset_pytorch.ToTensor()])

    positive_configs = dataset_configs.DatasetConfigs('train', beat_type, one_vs_all=True, lstm_setting=False,
                                                      over_sample_minority_class=False,
                                                      under_sample_majority_class=False,
                                                      only_take_heartbeat_of_type=beat_type, add_data_from_gan=False,
                                                      gan_configs=None)

    dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(positive_configs,
                                                             transform=composed)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)
    print("Size of real dataset is {}".format(len(dataset)))

    #
    # 2. Create the models:
    netG = generator_net.to(device)
    netD = discriminator_net.to(device)

    #
    # Define loss functions:
    #
    cross_entropy_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    #
    # Optimizers:
    #
    lr = 0.0002
    beta1 = 0.5
    writer.add_scalar('Learning_Rate', lr)
    optimizer_d = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    #
    # Noise for validation:
    #
    val_noise = torch.Tensor(np.random.normal(0, 1, (4, 100))).to(device)

    #
    # Training loop"
    #
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

            #
            # Discriminator from real beats:
            #
            ecg_batch = data['cardiac_cycle'].float().to(device)
            b_size = ecg_batch.shape[0]

            num_of_beats_seen += ecg_batch.shape[0]
            output = netD(ecg_batch)
            labels = torch.full((b_size,), 1, device=device)

            ce_loss_d_real = cross_entropy_loss(output, labels)
            writer.add_scalar('Discriminator/cross_entropy_on_real_batch', ce_loss_d_real.item(), global_step=iters)
            writer.add_scalars('Merged/losses', {'d_cross_entropy_on_real_batch': ce_loss_d_real.item()},
                               global_step=iters)
            ce_loss_d_real.backward()
            mean_d_real_output = output.mean().item()

            #
            # Discriminator from fake beats:
            #
            noise_input = torch.Tensor(np.random.normal(0, 1, (b_size, 100))).to(device)
            # noise_input = torch.Tensor(np.random.normal(0, 1, (b_size, 100))).to(device)

            output_g_fake = netG(noise_input)
            output = netD(output_g_fake.detach()).to(device)
            labels.fill_(0)

            ce_loss_d_fake = cross_entropy_loss(output, labels)
            writer.add_scalar('Discriminator/cross_entropy_on_fake_batch', ce_loss_d_fake.item(), iters)
            writer.add_scalars('Merged/losses', {'d_cross_entropy_on_fake_batch': ce_loss_d_fake.item()},
                               global_step=iters)
            ce_loss_d_fake.backward()

            mean_d_fake_output = output.mean().item()
            total_loss_d = ce_loss_d_fake + ce_loss_d_real
            writer.add_scalar(tag='Discriminator/total_loss', scalar_value=total_loss_d.item(),
                              global_step=iters)
            optimizer_d.step()

            netG.zero_grad()
            labels.fill_(1)
            output = netD(output_g_fake)

            #
            # Add euler loss:
            #
            delta_hb_signal, f_ode_z_signal = ode_loss(output_g_fake, ode_params, device, beat_type)
            mse_loss_euler = mse_loss(delta_hb_signal, f_ode_z_signal)
            logging.info("MSE ODE loss: {}".format(mse_loss_euler.item()))
            ce_loss_g_fake = cross_entropy_loss(output, labels)
            total_g_loss = mse_loss_euler + ce_loss_g_fake
            # total_g_loss = mse_loss_euler
            total_g_loss.backward()

            writer.add_scalar(tag='Generator/mse_ode', scalar_value=mse_loss_euler.item(), global_step=iters)
            writer.add_scalar(tag='Generator/cross_entropy_on_fake_batch', scalar_value=ce_loss_g_fake.item(),
                              global_step=iters)
            writer.add_scalars('Merged/losses', {'g_cross_entropy_on_fake_batch': ce_loss_g_fake.item()},
                               global_step=iters)
            mean_d_fake_output_2 = output.mean().item()

            optimizer_g.step()

            if iters % 50 == 0:
                print("{}/{}: Epoch #{}: Iteration #{}: Mean D(real_hb_batch) = {}, mean D(G(z)) = {}."
                      .format(num_of_beats_seen, len(dataset), epoch, iters, mean_d_real_output, mean_d_fake_output),
                      end=" ")
                print("mean D(G(z)) = {} After backprop of D".format(mean_d_fake_output_2))

                print("Loss D from real beats = {}. Loss D from Fake beats = {}. Total Loss D = {}".
                      format(ce_loss_d_real, ce_loss_d_fake, total_loss_d), end=" ")
                print("Loss G = {}".format(ce_loss_g_fake))

            #
            # Norma of gradients:
            #
            gNormGrad = get_gradient_norm_l2(netG)
            dNormGrad = get_gradient_norm_l2(netD)
            writer.add_scalar('Generator/gradients_norm', gNormGrad, iters)
            writer.add_scalar('Discriminator/gradients_norm', dNormGrad, iters)
            print(
                "Generator Norm of gradients = {}. Discriminator Norm of gradients = {}.".format(gNormGrad, dNormGrad))

            if iters % 25 == 0:
                with torch.no_grad():
                    with torch.no_grad():
                        netG.eval()
                        output_g = netG(val_noise)
                        netG.train()
                        fig = plt.figure()
                        plt.title("Fake beats from Generator. iteration {}".format(i))
                        for p in range(4):
                            plt.subplot(2, 2, p + 1)
                            plt.plot(output_g[p].cpu().detach().numpy(), label="fake beat")
                            plt.plot(ecg_batch[p].cpu().detach().numpy(), label="real beat")
                            plt.legend()
                        writer.add_figure('Generator/output_example', fig, iters)
                        plt.close()
            if iters % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': netG.state_dict(),
                }, model_dir + '/checkpoint_epoch_{}_iters_{}'.format(epoch, iters))
            iters += 1
        epoch += 1
    torch.save({
        'epoch': epoch,
        'generator_state_dict': netG.state_dict(),
    }, model_dir + '/checkpoint_epoch_{}_iters_{}'.format(epoch, iters))
    writer.close()


def get_gradient_norm_l2(model):
    total_norm = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'ode_params_generator' not in name:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    model_dir = args.MODEL_DIR
    gan_type = args.GAN_TYPE
    if gan_type == 'SimVGAN':
        netG = vanila_gan.VGenerator(0)
        netD = vanila_gan.VDiscriminator(0)
    elif gan_type == 'SimDCGAN':
        netG = sim_gan_euler.DCGenerator(0)
        netD = sim_gan_euler.DCDiscriminator(0)
        netD.apply(weights_init)
        netG.apply(weights_init)
    else:
        raise ValueError(f"Invalid gan type {gan_type}.")

    train(args.BATCH_SIZE, args.NUM_ITERATIONS, model_dir, beat_type=args.BEAT_TYPE, generator_net=netG,
          discriminator_net=netD)
