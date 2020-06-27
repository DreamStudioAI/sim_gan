import torch
import math
from sim_gan.dynamical_model import utils
import time
import matplotlib.pyplot as plt


class ODEParams:
    def __init__(self, device_name):
        self.A = torch.tensor(0.005).to(device_name)  # mV
        self.f1 = torch.tensor(0.1).to(device_name)  # mean 1
        self.f2 = torch.tensor(0.25).to(device_name)  # mean 2
        self.c1 = torch.tensor(0.01).to(device_name)  # std 1
        self.c2 = torch.tensor(0.01).to(device_name)  # std 2
        self.rrpc = utils.generate_omega_function(self.f1, self.f2, self.c1, self.c2)
        self.rrpc = torch.tensor(self.rrpc).to(device_name)
        self.h = torch.tensor(1 / 216).to(device_name)


def single_step_euler(ode_params, x_curr, y_curr, z_curr, t_curr, input_params,
                            device_name):

    h = ode_params.h
    A = ode_params.A
    f2 = ode_params.f2
    rrpc = ode_params.rrpc.float()

    a_p = input_params[0]
    a_q = input_params[3]
    a_r = input_params[6]
    a_s = input_params[9]
    a_t = input_params[12]

    b_p = input_params[1]
    b_q = input_params[4]
    b_r = input_params[7]
    b_s = input_params[10]
    b_t = input_params[13]

    theta_p = input_params[2]
    theta_q = input_params[5]
    theta_r = input_params[8]
    theta_s = input_params[11]
    theta_t = input_params[14]

    alpha = 1 - (x_curr * x_curr + y_curr * y_curr) ** 0.5
    cast = (t_curr / h).type(torch.IntTensor)
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    if rrpc[tensor_temp] == 0:
        print("***inside zero***")
        omega = (2.0 * math.pi / 1e-3)
        # omega = torch.tensor(math.inf).to(device_name)
    else:
        omega = (2.0 * math.pi / rrpc[tensor_temp]).to(device_name)



    d_x_d_t_next = alpha * x_curr - omega * y_curr

    d_y_d_t_next = alpha * y_curr + omega * x_curr

    theta = torch.atan2(y_curr, x_curr)
    delta_theta_p = torch.fmod(theta - theta_p, 2 * math.pi)
    delta_theta_q = torch.fmod(theta - theta_q, 2 * math.pi)
    delta_theta_r = torch.fmod(theta - theta_r, 2 * math.pi)
    delta_theta_s = torch.fmod(theta - theta_s, 2 * math.pi)
    delta_theta_t = torch.fmod(theta - theta_t, 2 * math.pi)

    z_p = a_p * delta_theta_p * \
           torch.exp((- delta_theta_p * delta_theta_p / (2 * b_p * b_p)))

    z_q = a_q * delta_theta_q * \
           torch.exp((- delta_theta_q * delta_theta_q / (2 * b_q * b_q)))

    z_r = a_r * delta_theta_r * \
           torch.exp((- delta_theta_r * delta_theta_r / (2 * b_r * b_r)))

    z_s = a_s * delta_theta_s * \
           torch.exp((- delta_theta_s * delta_theta_s / (2 * b_s * b_s)))

    z_t = a_t * delta_theta_t * \
          torch.exp((- delta_theta_t * delta_theta_t / (2 * b_t * b_t)))

    z_0_t = (A * torch.sin(torch.tensor(2 * math.pi).to(device_name) * f2 * t_curr).to(device_name)).to(device_name)

    d_z_d_t_next = -1 * (z_p + z_q + z_r + z_s + z_t) - (z_curr - z_0_t)

    k1_x = h * d_x_d_t_next

    k1_y = h * d_y_d_t_next

    k1_z = h * d_z_d_t_next
    # Calculate next stage:
    x_next = x_curr + k1_x
    y_next = y_curr + k1_y
    z_next = z_curr + k1_z

    return x_next, y_next, z_next


if __name__ == "__main__":

    ode_params = ODEParams('cpu')

    input_params = torch.nn.Parameter(
        torch.tensor([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
                      90.0 * math.pi / 180.0]))
    x = torch.tensor(-0.417750770388669)
    y = torch.tensor(-0.9085616622823985)
    z = torch.tensor(-0.004551233843726818)
    # x = torch.tensor(1.0)
    # y = torch.tensor(0.0)
    # z = torch.tensor(0.04)
    t = torch.tensor(0.0)
    x_next, y_next, z_next = single_step_euler(ode_params, x, y, z, t, input_params, 'cpu')
    x_t = [x_next]
    y_t = [y_next]
    z_t = [z_next]
    start = time.time()
    for i in range(215 * 1):
        last = z_t[-1]

        t += 1 / 512
        x_next, y_next, z_next = single_step_euler(ode_params, x_next, y_next, z_next, t, input_params, 'cpu')
        x_t.append(x_next)
        y_t.append(y_next)
        z_t.append(z_next)
    end = time.time()
    print("time: ", end - start)
    # last = z_t[-1]
    # print(last.backward())
    # print(input_params.grad)
    print("Z: {}".format([x.detach().item() for x in z_t]))
    print("X: {}".format([x.detach().item() for x in x_t]))
    print("Y: {}".format([x.detach().item() for x in y_t]))
    print(len(z_t))
    print("Max value in the signal: ", max(z_t))
    print("Min valuein the signal:", min(z_t))
    res = [x.detach().numpy() for x in z_t]
    print(len(res))
    plt.plot(res)
    plt.show()
