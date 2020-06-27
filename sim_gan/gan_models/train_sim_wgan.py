"""This modoule trains a WGAN with additional Euler loss."""
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from sim_gan.data_reader import ecg_dataset_pytorch
from tensorboardX import SummaryWriter
from sim_gan.gan_models.models import wgan
import logging
from sim_gan.data_reader import dataset_configs
from sim_gan.dynamical_model import equations
from sim_gan.dynamical_model.ode_params import ODEParams
import torchvision.transforms as transforms
from sim_gan.dynamical_model import typical_beat_params
import math


TYPICAL_ODE_N_PARAMS = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                      160.0 * math.pi / 180.0]


def generate_typical_N_ode_params(b_size, device):
    noise_param = torch.from_numpy(np.random.normal(0, 0.1, (b_size, 15))).to(device)
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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(batch_size, num_train_steps, generator, discriminator, model_dir, beat_type, device):

    ode_params = ODEParams(device)
    #
    # Support for tensorboard:
    #
    writer = SummaryWriter(model_dir)

    #
    # 1. create the ECG dataset:
    #
    positive_configs = dataset_configs.DatasetConfigs('train', beat_type, one_vs_all=True, lstm_setting=False,
                                                      over_sample_minority_class=False,
                                                      under_sample_majority_class=False,
                                                      only_take_heartbeat_of_type=beat_type, add_data_from_gan=False,
                                                      gan_configs=None)

    composed = transforms.Compose([ecg_dataset_pytorch.Scale(), ecg_dataset_pytorch.ToTensor()])
    dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(positive_configs,
                                                             transform=composed)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)
    print("Size of real dataset is {}".format(len(dataset)))

    #
    # 2. Create the Networks:
    #
    netG = generator.float()
    netD = discriminator.float()

    num_d_iters = 5
    weight_cliping_limit = 0.01
    #
    # Loss functions for WGAN and ode:
    #
    mse_loss = nn.MSELoss()

    # Optimizers:
    # WGAN values from paper
    lr = 0.00005


    writer.add_scalar('Learning_Rate', lr)
    # WGAN with gradient clipping uses RMSprop instead of ADAM
    optimizer_d = torch.optim.RMSprop(netD.parameters(), lr=lr)
    optimizer_g = torch.optim.RMSprop(netG.parameters(), lr=lr)

    # Noise for validation:
    val_noise = torch.from_numpy(np.random.uniform(0, 1, (4, 100))).float().to(device)
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

            # Train Dicriminator forward - loss - backward - update num_d_iters times while 1 Generator
            # forward-loss-backward-update
            for p in netD.parameters():
                p.requires_grad = True
            for d_iter in range(num_d_iters):

                netD.zero_grad()

                # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                for p in netD.parameters():
                    p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

                ecg_batch = data['cardiac_cycle'].float().to(device)
                b_size = ecg_batch.shape[0]

                # Check for batch to have full batch_size
                if (b_size != batch_size):
                    continue
                num_of_beats_seen += ecg_batch.shape[0]

                output = netD(ecg_batch)

                # Adversarial loss
                loss_d_real = -torch.mean(output)

                writer.add_scalar('Discriminator/cross_entropy_on_real_batch', loss_d_real.item(), global_step=iters)
                writer.add_scalars('Merged/losses', {'d_cross_entropy_on_real_batch': loss_d_real.item()},
                                   global_step=iters)
                loss_d_real.backward()
                loss_d_real_hist.append(loss_d_real.item())

                mean_d_real_output = output.mean().item()
                d_real_pred_hist.append(mean_d_real_output)

                #
                # D loss from fake:
                #
                noise_input = torch.from_numpy(np.random.uniform(0, 1, (b_size, 100))).float().to(device)

                output_g_fake = netG(noise_input)
                output = netD(output_g_fake.detach())

                loss_d_fake = torch.mean(output)
                # ce_loss_d_fake = cross_entropy_loss(output, labels)
                writer.add_scalar('Discriminator/cross_entropy_on_fake_batch', loss_d_fake.item(), iters)
                writer.add_scalars('Merged/losses', {'d_cross_entropy_on_fake_batch': loss_d_fake.item()},
                                   global_step=iters)
                loss_d_fake.backward()

                loss_d_fake_hist.append(loss_d_fake.item())

                mean_d_fake_output = output.mean().item()
                d_fake_pred_hist.append(mean_d_fake_output)
                total_loss_d = loss_d_fake + loss_d_real
                writer.add_scalar(tag='Discriminator/total_loss', scalar_value=total_loss_d.item(),
                                  global_step=iters)
                optimizer_d.step()

            #
            # Generator updates:
            #
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation

            netG.zero_grad()

            noise_input = torch.from_numpy(np.random.uniform(0, 1, (batch_size, 100))).float().to(device)

            output_g_fake = netG(noise_input)

            output = netD(output_g_fake)

            # Adversarial loss:
            loss_g_fake = -torch.mean(output)

            # Euler loss:
            delta_hb_signal, f_ode_z_signal = ode_loss(output_g_fake, ode_params, device, beat_type)
            mse_loss_euler = mse_loss(delta_hb_signal, f_ode_z_signal)
            logging.info("MSE ODE loss: {}".format(mse_loss_euler.item()))

            total_g_loss = mse_loss_euler + loss_g_fake

            total_g_loss.backward()
            loss_g_fake_hist.append(loss_g_fake.item())
            writer.add_scalar(tag='Generator/mse_ode', scalar_value=mse_loss_euler.item(), global_step=iters)
            writer.add_scalar(tag='Generator/cross_entropy_on_fake_batch', scalar_value=loss_g_fake.item(),
                              global_step=iters)
            writer.add_scalars('Merged/losses', {'g_cross_entropy_on_fake_batch': total_g_loss.item()},
                               global_step=iters)
            mean_d_fake_output_2 = output.mean().item()

            optimizer_g.step()

            print("{}/{}: Epoch #{}: Iteration #{}: Mean D(real_hb_batch) = {}, mean D(G(z)) = {}."
                  .format(num_of_beats_seen, len(dataset), epoch, iters, mean_d_real_output, mean_d_fake_output),
                  end=" ")
            print("mean D(G(z)) = {} After backprop of D".format(mean_d_fake_output_2))

            print("Loss D from real beats = {}. Loss D from Fake beats = {}. Total Loss D = {}".
                  format(loss_d_real, loss_d_fake, total_loss_d), end=" ")
            print("Loss G = {}".format(loss_g_fake))

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
                        plt.plot(output_g[p].cpu().detach().numpy(), label="fake beat")
                        plt.plot(ecg_batch[p].cpu().detach().numpy(), label="real beat")
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    netG = wgan.WGenerator(0).to(device)
    netG.apply(weights_init)
    netD = wgan.WDiscriminator(0).to(device)
    netD.apply(weights_init)
    model_dir = 'tensorboard/wgan_ode_F_beat'
    train(batch_size=64, num_train_steps=2000, generator=netG, discriminator=netD, model_dir=model_dir, beat_type='F',
          device=device)
