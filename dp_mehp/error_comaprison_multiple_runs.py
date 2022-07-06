# in this script, we illustrate the approximation error to the Gaussian kernel using different features
# features of interest: (1) Random Fourier features; and (2) Hermite Polynomials (from the Mehler formula)

import numpy as np
import kernel as k
from all_aux_files import meddistance
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from all_aux_files import find_rho


def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    W = torch.Tensor(W)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
    return Z


def phi_recursion(phi_k, phi_k_minus_1, rho, degree, x_in):
  if degree == 0:
    phi_0 = (1 - rho) ** (0.25) * (1 + rho) ** (0.25) * torch.exp(-rho / (1 + rho) * x_in ** 2)
    return phi_0
  elif degree == 1:
    phi_1 = np.sqrt(2 * rho) * x_in * phi_k
    return phi_1
  else:  # from degree ==2 (k=1 in the recursion formula)
    k = degree - 1
    first_term = np.sqrt(rho) / np.sqrt(2 * (k + 1)) * 2 * x_in * phi_k
    second_term = rho / np.sqrt(k * (k + 1)) * k * phi_k_minus_1
    phi_k_plus_one = first_term - second_term
    return phi_k_plus_one


def compute_phi(x_in, n_degrees, rho, device):
  first_dim = x_in.shape[0]
  batch_embedding = torch.empty(first_dim, n_degrees, dtype=torch.float32, device=device)
  # batch_embedding = torch.zeros(first_dim, n_degrees).to(device)
  phi_i_minus_one, phi_i_minus_two = None, None
  for degree in range(n_degrees):
    phi_i = phi_recursion(phi_i_minus_one, phi_i_minus_two, rho, degree, x_in.squeeze())
    batch_embedding[:, degree] = phi_i

    phi_i_minus_two = phi_i_minus_one
    phi_i_minus_one = phi_i

  return batch_embedding


def feature_map_HP(k, x, rho, device):
  # k: degree of polynomial
  # rho: a parameter (related to length parameter)
  # x: where to evaluate the function at

  eigen_vals = (1 - rho) * (rho ** torch.arange(0, k + 1))
  eigen_vals = eigen_vals.to(device)
  phi_x = compute_phi(x, k + 1, rho, device)

  return phi_x, eigen_vals


def ME_with_HP(x, order, rho, device, n_training_data):
  n_data, input_dim = x.shape

  # reshape x, such that x is a long vector
  x_flattened = x.view(-1)
  x_flattened = x_flattened[:, None]
  phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_HP(order, x_flattened, rho, device)
  phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order + 1)
  phi_x = phi_x.type(torch.float)

  return phi_x

def err(A,B):
    return torch.norm(A-B)

def main():
    # generate some data from a Gaussian distribution
    input_dim = 1
    n_data = 100
    runs=100
    # feature_dim = 5
    max_order = 500
    err_RF = np.zeros((max_order, runs))
    err_HP = np.zeros((max_order))
    device = 'cpu'
    mean = 0
    mean_prime = 1

    x = mean + np.random.randn(n_data,1)
    x = np.sort(x, axis=0)

      
    x_prime = mean_prime + np.random.randn(n_data,1)
    x_prime = np.sort(x_prime, axis=0)

    # evaluate the kernel function
    med = meddistance(np.concatenate((x,x_prime),axis=0))
    sigma2 = med**2
    X, Y = np.meshgrid(x, x_prime)
    D2 = (X - Y)**2
    K = np.exp(-D2 / (2.0 * sigma2))

    """ first we plot the features evaluated at x """

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)

    for i in range(max_order):
      print('# of features', i+1)

      for run in range(0, runs):
        np.random.seed(run)

        n_features = i + 1  # so the order is from 0 to 1001
        draws = n_features // 2
        W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)
        emb1 = RFF_Gauss(n_features, torch.Tensor(x), W_freq) #Add noise
        emb2 = RFF_Gauss(n_features, torch.Tensor(x_prime), W_freq)
        RF = torch.matmul(emb2, emb1.transpose(1, 0))
        err_RF[i, run] = err(torch.Tensor(K), RF) 

      rho = find_rho(sigma2, False)
      order = i + 1
      phi_1 = ME_with_HP(torch.Tensor(x), order, rho, device, n_data) #Add noise
      phi_2 = ME_with_HP(torch.Tensor(x_prime), order, rho, device, n_data)
      HP = torch.matmul(phi_2.squeeze(1), phi_1.squeeze(1).transpose(1, 0))
      err_HP[i] = err(torch.Tensor(K), HP)

    #print("This is err_RF matrix: ", err_RF)
    mean_err_RF=np.mean(err_RF, axis=1)
    #std_err_RF=np.std(err_RF, axis=1)


    #To save results.
    #np.save("err_RF_seed_100.npy", err_RF)
    #np.save("err_HP_seed_100.npy", err_HP)

    """Plotting"""
    plt.figure(2)
    plt.subplot(212)
    plt.plot(np.arange(0, max_order), mean_err_RF, 'o-', linewidth=3.0)
    #plt.errorbar(np.arange(0, max_order), mean_err_RF, std_err_RF,  fmt='o-')
    plt.plot(np.arange(0, max_order), np.min(mean_err_RF)*np.ones(max_order), 'k--')
    plt.title('error from RF approximation')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('number of random features')
    plt.subplot(211)
    plt.plot(np.arange(0, max_order), err_HP, 'o-', linewidth=3.0)
    plt.plot(np.arange(0, max_order), np.min(mean_err_RF) * np.ones(max_order), 'k--')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('error from HP approximation')
    plt.xlabel('order of polynomials')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
