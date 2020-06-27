import torch
import logging
import math
from sim_gan.dynamical_model.ode_params import ODEParams, ODEParamsNumpy
import time
from matplotlib import pyplot as plt
import autograd.numpy as np


def d_x_d_t(y, x, t, rrpc, delta_t):
    alpha = 1 - ((x * x) + (y * y)) ** 0.5

    cast = (t / delta_t).type(torch.IntTensor)
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    if rrpc[tensor_temp] == 0:
        logging.info("***inside zero***")
        omega = (2.0 * math.pi / 1e-3)
    else:
        omega = (2.0 * math.pi / rrpc[tensor_temp])

    f_x = alpha * x - omega * y
    return f_x


def d_x_d_t_batches(y, x, t, rrpc, delta_t):
    """

    :param y: [N]
    :param x: [N]
    :param t: [N]
    :param rrpc:
    :param delta_t:
    :return:
    """
    alpha = 1 - ((x * x) + (y * y)) ** 0.5
    cast = (t / delta_t).long()
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    specific_rrpc_values = rrpc[tensor_temp]
    omega = torch.zeros_like(x).float()
    zero_indexes = (specific_rrpc_values == 0).nonzero()[:, 0]
    non_zero_indexes = (specific_rrpc_values != 0).nonzero()[:, 0]
    omega[zero_indexes] = (2.0 * math.pi / 1e-3)
    omega[non_zero_indexes] = (2.0 * math.pi / specific_rrpc_values[non_zero_indexes])
    f_x = alpha * x - omega * y
    return f_x


def d_x_d_t_numpy(y, x, t, rrpc, delta_t):
    alpha = 1 - ((x * x) + (y * y)) ** 0.5

    cast = int(t / delta_t)
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    if rrpc[tensor_temp] == 0:
        logging.info("***inside zero***")
        omega = (2.0 * math.pi / 1e-3)
    else:
        omega = (2.0 * math.pi / rrpc[tensor_temp])

    f_x = alpha * x - omega * y
    return f_x


def d_x_d_t_numpy_batchs(y, x, t, rrpc, delta_t):
  alpha = 1 - ((x * x) + (y * y)) ** 0.5
  cast = (t / delta_t).astype(int)
  tensor_temp = 1 + cast
  tensor_temp = tensor_temp % len(rrpc)

  specific_rrpc_values = rrpc[tensor_temp]

  omega = np.zeros_like(x).astype(float)

  zero_indexes = np.argwhere(specific_rrpc_values == 0)[:, 0]
  non_zero_indexes = np.argwhere(specific_rrpc_values != 0)[:, 0]

  omega[zero_indexes] = (2.0 * math.pi / 1e-3)
  omega[non_zero_indexes] = (2.0 * math.pi / specific_rrpc_values[non_zero_indexes])
  f_x = alpha * x - omega * y
  return f_x


def d_y_d_t(y, x, t, rrpc, delta_t):
    alpha = 1 - ((x * x) + (y * y)) ** 0.5

    cast = (t / delta_t).type(torch.IntTensor)
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    if rrpc[tensor_temp] == 0:
        logging.info("***inside zero***")
        omega = (2.0 * math.pi / 1e-3)
    else:
        omega = (2.0 * math.pi / rrpc[tensor_temp])

    f_y = alpha * y + omega * x
    return f_y


def d_y_d_t_batches(y, x, t, rrpc, delta_t):
    alpha = 1 - ((x * x) + (y * y)) ** 0.5
    cast = (t / delta_t).long()
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    specific_rrpc_values = rrpc[tensor_temp]
    omega = torch.zeros_like(x).float()
    zero_indexes = (specific_rrpc_values == 0).nonzero()[:, 0]
    non_zero_indexes = (specific_rrpc_values != 0).nonzero()[:, 0]
    omega[zero_indexes] = (2.0 * math.pi / 1e-3)
    omega[non_zero_indexes] = (2.0 * math.pi / specific_rrpc_values[non_zero_indexes])
    f_y = alpha * x + omega * y
    return f_y


def d_y_d_t_numpy(y, x, t, rrpc, delta_t):
    alpha = 1 - ((x * x) + (y * y)) ** 0.5

    cast = int(t / delta_t)
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    if rrpc[tensor_temp] == 0:
        logging.info("***inside zero***")
        omega = (2.0 * math.pi / 1e-3)
    else:
        omega = (2.0 * math.pi / rrpc[tensor_temp])

    f_y = alpha * y + omega * x
    return f_y


def d_y_d_t_numpy_batchs(y, x, t, rrpc, delta_t):
    alpha = 1 - ((x * x) + (y * y)) ** 0.5

    cast = (t / delta_t).astype(int)
    tensor_temp = 1 + cast
    tensor_temp = tensor_temp % len(rrpc)
    specific_rrpc_values = rrpc[tensor_temp]

    omega = np.zeros_like(x).astype(float)

    zero_indexes = np.argwhere(specific_rrpc_values == 0)[:, 0]
    non_zero_indexes = np.argwhere(specific_rrpc_values != 0)[:, 0]

    omega[zero_indexes] = (2.0 * math.pi / 1e-3)
    omega[non_zero_indexes] = (2.0 * math.pi / specific_rrpc_values[non_zero_indexes])

    f_y = alpha * y + omega * x
    return f_y


def d_z_d_t(x, y, z, t, params, ode_params):
    """

    :param x:
    :param y:
    :param z:
    :param t:
    :param params:
    :param ode_params: Nx15
    :return:
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    A = ode_params.A
    f2 = ode_params.f2
    a_p, a_q, a_r, a_s, a_t = params[:, 0], params[:, 3], params[:, 6], params[:, 9], params[:, 12]

    b_p, b_q, b_r, b_s, b_t = params[:, 1], params[:, 4], params[:, 7], params[:, 10], params[:, 13]

    theta_p, theta_q, theta_r, theta_s, theta_t = params[:, 2], params[:, 5], params[:, 8], params[:, 11], params[:, 14]

    a_p = a_p.view(-1, 1)
    a_q = a_q.view(-1, 1)
    a_r = a_r.view(-1, 1)
    a_s = a_s.view(-1, 1)
    a_t = a_t.view(-1, 1)

    b_p = b_p.view(-1, 1)
    b_q = b_q.view(-1, 1)
    b_r = b_r.view(-1, 1)
    b_s = b_s.view(-1, 1)
    b_t = b_t.view(-1, 1)

    theta_p = theta_p.view(-1, 1)
    theta_q = theta_q.view(-1, 1)
    theta_r = theta_r.view(-1, 1)
    theta_s = theta_s.view(-1, 1)
    theta_t = theta_t.view(-1, 1)

    logging.debug("theta p shape: {}".format(theta_p.shape))
    theta = torch.atan2(y, x)
    logging.debug("theta shape: {}".format(theta.shape))
    logging.debug("delta before mod: {}".format((theta - theta_p).shape))
    delta_theta_p = torch.fmod(theta - theta_p, 2 * math.pi)
    logging.debug("delta theta shape: {}".format(delta_theta_p.shape))
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

    z_0_t = (A * torch.sin(2 * math.pi * f2 * t))

    z_p = z_p.to(device)
    z_q = z_q.to(device)
    z_r = z_r.to(device)
    z_s = z_s.to(device)
    z_t = z_t.to(device)
    z_0_t = z_0_t.to(device)

    f_z = -1 * (z_p + z_q + z_r + z_s + z_t) - (z - z_0_t)
    return f_z


def d_z_d_t_numpy(x, y, z, t, params, ode_params):
    A = ode_params.A
    f2 = ode_params.f2
    a_p, a_q, a_r, a_s, a_t = params[:, 0], params[:, 3], params[:, 6], params[:, 9], params[:, 12]

    b_p, b_q, b_r, b_s, b_t = params[:, 1], params[:, 4], params[:, 7], params[:, 10], params[:, 13]

    theta_p, theta_q, theta_r, theta_s, theta_t = params[:, 2], params[:, 5], params[:, 8], params[:, 11], params[:, 14]

    a_p = a_p.reshape((-1, 1))
    a_q = a_q.reshape((-1, 1))
    a_r = a_r.reshape((-1, 1))
    a_s = a_s.reshape((-1, 1))
    a_t = a_t.reshape((-1, 1))

    b_p = b_p.reshape((-1, 1))
    b_q = b_q.reshape((-1, 1))
    b_r = b_r.reshape((-1, 1))
    b_s = b_s.reshape((-1, 1))
    b_t = b_t.reshape((-1, 1))

    theta_p = theta_p.reshape((-1, 1))
    theta_q = theta_q.reshape((-1, 1))
    theta_r = theta_r.reshape((-1, 1))
    theta_s = theta_s.reshape((-1, 1))
    theta_t = theta_t.reshape((-1, 1))

    logging.debug("theta p shape: {}".format(theta_p.shape))
    theta = np.arctan2(y, x)
    logging.debug("theta shape: {}".format(theta.shape))
    logging.debug("delta before mod: {}".format((theta - theta_p).shape))
    delta_theta_p = np.fmod(theta - theta_p, 2 * math.pi)
    logging.debug("delta theta shape: {}".format(delta_theta_p.shape))
    delta_theta_q = np.fmod(theta - theta_q, 2 * math.pi)
    delta_theta_r = np.fmod(theta - theta_r, 2 * math.pi)
    delta_theta_s = np.fmod(theta - theta_s, 2 * math.pi)
    delta_theta_t = np.fmod(theta - theta_t, 2 * math.pi)

    z_p = a_p * delta_theta_p * \
          np.exp((- delta_theta_p * delta_theta_p / (2 * b_p * b_p)))

    z_q = a_q * delta_theta_q * \
          np.exp((- delta_theta_q * delta_theta_q / (2 * b_q * b_q)))

    z_r = a_r * delta_theta_r * \
          np.exp((- delta_theta_r * delta_theta_r / (2 * b_r * b_r)))

    z_s = a_s * delta_theta_s * \
          np.exp((- delta_theta_s * delta_theta_s / (2 * b_s * b_s)))

    z_t = a_t * delta_theta_t * \
          np.exp((- delta_theta_t * delta_theta_t / (2 * b_t * b_t)))

    z_0_t = (A * np.sin(2 * math.pi * f2 * t))

    z_p = z_p
    z_q = z_q
    z_r = z_r
    z_s = z_s
    z_t = z_t
    z_0_t = z_0_t

    f_z = -1 * (z_p + z_q + z_r + z_s + z_t) - (z - z_0_t)
    return f_z


def d_z_d_t_numpy_batchs(x, y, z, t, params, ode_params):
  a_p, a_q, a_r, a_s, a_t = params[0], params[3], params[6], params[9], params[12]

  b_p, b_q, b_r, b_s, b_t = params[1], params[4], params[7], params[10], params[13]

  theta_p, theta_q, theta_r, theta_s, theta_t = params[2], params[5], params[8], params[11], params[14]

  theta = np.arctan2(y, x)
  delta_theta_p = np.fmod(theta - theta_p, 2 * math.pi)
  delta_theta_q = np.fmod(theta - theta_q, 2 * math.pi)
  delta_theta_r = np.fmod(theta - theta_r, 2 * math.pi)
  delta_theta_s = np.fmod(theta - theta_s, 2 * math.pi)
  delta_theta_t = np.fmod(theta - theta_t, 2 * math.pi)

  z_p = a_p * delta_theta_p * \
      np.exp((- delta_theta_p * delta_theta_p / (2 * b_p * b_p)))

  z_q = a_q * delta_theta_q * \
        np.exp((- delta_theta_q * delta_theta_q / (2 * b_q * b_q)))

  z_r = a_r * delta_theta_r * \
        np.exp((- delta_theta_r * delta_theta_r / (2 * b_r * b_r)))

  z_s = a_s * delta_theta_s * \
        np.exp((- delta_theta_s * delta_theta_s / (2 * b_s * b_s)))

  z_t = a_t * delta_theta_t * \
        np.exp((- delta_theta_t * delta_theta_t / (2 * b_t * b_t)))

  z_0_t = (ode_params.A * np.sin(2 * math.pi * ode_params.f2 * t))

  f_z = -1 * (z_p + z_q + z_r + z_s + z_t) - (z - z_0_t)
  return f_z


def d_z_d_t_batches(x, y, z, t, params, ode_params):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    A = ode_params.A
    f2 = ode_params.f2
    a_p, a_q, a_r, a_s, a_t = params[0], params[3], params[6], params[9], params[12]

    b_p, b_q, b_r, b_s, b_t = params[1], params[4], params[7], params[10], params[13]

    theta_p, theta_q, theta_r, theta_s, theta_t = params[2], params[5], params[8], params[11], params[14]

    # a_p = a_p.view(-1, 1)
    # a_q = a_q.view(-1, 1)
    # a_r = a_r.view(-1, 1)
    # a_s = a_s.view(-1, 1)
    # a_t = a_t.view(-1, 1)
    #
    # b_p = b_p.view(-1, 1)
    # b_q = b_q.view(-1, 1)
    # b_r = b_r.view(-1, 1)
    # b_s = b_s.view(-1, 1)
    # b_t = b_t.view(-1, 1)
    #
    # theta_p = theta_p.view(-1, 1)
    # theta_q = theta_q.view(-1, 1)
    # theta_r = theta_r.view(-1, 1)
    # theta_s = theta_s.view(-1, 1)
    # theta_t = theta_t.view(-1, 1)

    logging.debug("theta p shape: {}".format(theta_p.shape))
    theta = torch.atan2(y, x)
    logging.debug("theta shape: {}".format(theta.shape))
    logging.debug("delta before mod: {}".format((theta - theta_p).shape))
    delta_theta_p = torch.fmod(theta - theta_p, 2 * math.pi)
    logging.debug("delta theta shape: {}".format(delta_theta_p.shape))
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

    z_0_t = (A * torch.sin(2 * math.pi * f2 * t))

    z_p = z_p.to(device)
    z_q = z_q.to(device)
    z_r = z_r.to(device)
    z_s = z_s.to(device)
    z_t = z_t.to(device)
    z_0_t = z_0_t.to(device)

    f_z = -1 * (z_p + z_q + z_r + z_s + z_t) - (z - z_0_t)
    return f_z


def generate_batch_of_beats_numpy(params):
    ode_params = ODEParamsNumpy()
    x = np.array([-0.417750770388669 for _ in range(params.shape[0])]).reshape((-1, 1))
    y = np.array([-0.9085616622823985 for _ in range(params.shape[0])]).reshape((-1, 1))
    z = np.array([-0.004551233843726818 for _ in range(params.shape[0])]).reshape((-1, 1))
    t = 0.0

    x_signal = [x]
    y_signal = [y]
    z_signal = [z]
    start = time.time()
    for i in range(215):
        f_x = d_x_d_t_numpy(y, x, t, ode_params.rrpc, ode_params.h)
        f_y = d_y_d_t_numpy(y, x, t, ode_params.rrpc, ode_params.h)
        f_z = d_z_d_t_numpy(x, y, z, t, params, ode_params)

        t += 1 / 512
        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z
    end = time.time()
    logging.info("Time to generate batch: {}".format(end - start))
    z_signal = np.stack(z_signal).reshape((216, -1)).transpose()
    return z_signal


def test_equations():
    ode_params = ODEParams('cpu')

    input_params = torch.nn.Parameter(
        torch.tensor([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
                      90.0 * math.pi / 180.0])).view(1, 15)
    x = torch.tensor(-0.417750770388669)
    y = torch.tensor(-0.9085616622823985)
    z = torch.tensor(-0.004551233843726818)
    t = torch.tensor(0.0)

    x_signal = [x]
    y_signal = [y]
    z_signal = [z]
    start = time.time()
    for i in range(215):
        f_x = d_x_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_y = d_y_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_z = d_z_d_t(x, y, z, t, input_params, ode_params)

        t += 1 / 512
        logging.debug("f_z shape: {}".format(f_z.shape))
        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z
    end = time.time()
    logging.info("time: {}".format(end - start))

    res = [v.detach().numpy() for v in z_signal]

    print(len(res))
    plt.plot(res)
    plt.show()

    # logging.DEBUG("Z: {}".format([x.detach().item() for x in z_signal]))
    # logging.DEBUG("X: {}".format([x.detach().item() for x in x_signal]))
    # logging.DEBUG("Y: {}".format([x.detach().item() for x in y_signal]))


def test_equations_on_batch():
    ode_params = ODEParams('cpu')

    input = np.array((
             ([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
                           30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
                           90.0 * math.pi / 180.0]))).reshape((1,15))
    a1 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    a2 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    print(a1.shape)
    input_params = torch.cat((a1, a2), 0)
    print("Input params shape: {}".format(input_params.shape))
    x = torch.tensor([-0.417750770388669, -0.417750770388669]).view(2, 1)
    y = torch.tensor([-0.9085616622823985, -0.9085616622823985]).view(2, 1)
    z = torch.tensor([-0.004551233843726818, 0.03]).view(2, 1)
    t = torch.tensor(0.0)

    x_signal = [x]
    y_signal = [y]
    z_signal = [z]
    start = time.time()
    for i in range(215):
        f_x = d_x_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_y = d_y_d_t(y, x, t, ode_params.rrpc, ode_params.h)
        f_z = d_z_d_t(x, y, z, t, input_params, ode_params)
        t += 1 / 512
        logging.info("f_z shape: {}".format(f_z.shape))
        logging.info("f_y shape: {}".format(f_y.shape))
        logging.info("f_x shape: {}".format(f_x.shape))
        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z
    end = time.time()

    logging.info("time: {}".format(end - start))
    z_signal = torch.stack(z_signal)
    logging.info('z_signal shape: {}'.format(z_signal.shape))
    res = [v[0].detach().numpy() for v in z_signal]

    print(len(res))
    print(res[0])
    plt.plot(res)
    plt.show()

    res = [v[1].detach().numpy() for v in z_signal]

    print(len(res))
    print(res[0])
    plt.plot(res)
    plt.show()
    print(z_signal[:10])
    print(x_signal[:10])
    print(y_signal[:10])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_equations()
    # test_equations_on_batch()

    # x = -0.417750770388669
    # y = -0.9085616622823985
    # z = -0.004551233843726818
    # t = 0.0
    # ode_params = ODEParamsNumpy()
    # print(d_x_d_t_numpy(y, x, t, ode_params.rrpc, delta_t=1.0))

