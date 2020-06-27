from sim_gan.dynamical_model.Euler.single_step import single_step_euler
import torch
import torch.nn as nn
from sim_gan.dynamical_model.ode_params import ODEParams


class Euler(nn.Module):
    def __init__(self, device_name):
        super(Euler, self).__init__()
        self.device_name = device_name

    def forward(self, x, v0):
        x = euler(x, self.device_name, v0)
        x = scale_signal(x)
        return x


def scale_signal(ecg_signal, min_val=-0.01563, max_val=0.042557):
    res = []
    for beat in ecg_signal:
        # Scale signal to lie between -0.4 and 1.2 mV :
        zmin = min(beat)
        zmax = max(beat)
        zrange = zmax - zmin
        scaled = [(z - zmin) * max_val / zrange + min_val for z in beat]
        scaled = torch.stack(scaled)
        res.append(scaled)
    res = torch.stack(res)
    return res


def down_sample(ecg_signal):
    res = []
    for beat in ecg_signal:
        i = 0
        down_sampled_ecg = []
        q = int(514 / 216)
        while i < 514:
            # j += 1
            if len(down_sampled_ecg) == 216:
                break
            down_sampled_ecg.append(beat[i])
            i += q  # q = int(np.rint(self.ecg_params.getSf() / self.ecg_params.getSfEcg()))
        down_sampled_ecg = torch.stack(down_sampled_ecg)
        res.append(down_sampled_ecg)
    res = torch.stack(res)
    return res


def euler(params_batch, device_name, v0):
    ode_params = ODEParams(device_name)
    x = torch.tensor(-0.417750770388669).to(device_name)
    y = torch.tensor(-0.9085616622823985).to(device_name)

    res = []
    for j, params in enumerate(params_batch):
        z = torch.tensor(v0[j]).to(device_name)
        t = torch.tensor(0.0).to(device_name)
        x_next, y_next, z_next = single_step_euler(ode_params, x, y, z, t, params, device_name)
        x_t = [x_next]
        y_t = [y_next]
        z_t = [z_next]
        for i in range(215):
            t += 1 / 216
            x_next, y_next, z_next = single_step_euler(ode_params, x_next, y_next, z_next, t, params, device_name)
            x_t.append(x_next)
            y_t.append(y_next)
            z_t.append(z_next)
        z_t = torch.stack(z_t)
        res.append(z_t)
    res = torch.stack(res)
    return res
