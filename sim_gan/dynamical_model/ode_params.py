import torch
from sim_gan.dynamical_model import utils


class ODEParams:
    def __init__(self, device_name):
        self.A = torch.tensor(0.005).to(device_name)  # mV
        self.f1 = torch.tensor(0.1).to(device_name)  # mean 1
        self.f2 = torch.tensor(0.25).to(device_name)  # mean 2
        self.c1 = torch.tensor(0.01).to(device_name)  # std 1
        self.c2 = torch.tensor(0.01).to(device_name)  # std 2
        self.rrpc = utils.generate_omega_function(self.f1, self.f2, self.c1, self.c2)
        self.rrpc = torch.tensor(self.rrpc).to(device_name).float()
        self.h = torch.tensor(1 / 216).to(device_name)


class ODEParamsNumpy:
    def __init__(self):
        self.A = 0.005  # mV
        self.f1 = 0.1  # mean 1
        self.f2 = 0.25  # mean 2
        self.c1 = 0.01  # std 1
        self.c2 = 0.01  # std 2
        self.rrpc = utils.generate_omega_function(self.f1, self.f2, self.c1, self.c2)
        self.h = 1 / 216
