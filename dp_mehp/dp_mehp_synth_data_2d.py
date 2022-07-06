import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from all_aux_files import log_args
from autodp import privacy_calibrator
import argparse
from all_aux_files_tab_data import data_loading
from all_aux_files_tab_data import test_models, ME_with_HP_tab, find_rho_tab, ME_with_HP_tab_combined_k, heuristic_for_length_scale
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from all_aux_files import ME_with_HP_prod, find_rho, synthesize_data_with_uniform_labels, ME_with_HP

import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import sys

class FCCondGen(nn.Module):
    def __init__(self, d_code, d_hid, d_out, n_labels, use_sigmoid=True, batch_norm=True):
        super(FCCondGen, self).__init__()
        d_hid = [int(k) for k in d_hid.split(',')]
        assert len(d_hid) < 5

        self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
        self.fc2 = nn.Linear(d_hid[0], d_hid[1])

        self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
        if len(d_hid) == 2:
            self.fc3 = nn.Linear(d_hid[1], d_out)
        elif len(d_hid) == 3:
            self.fc3 = nn.Linear(d_hid[1], d_hid[2])
            self.fc4 = nn.Linear(d_hid[2], d_out)
            self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
        elif len(d_hid) == 4:
            self.fc3 = nn.Linear(d_hid[1], d_hid[2])
            self.fc4 = nn.Linear(d_hid[2], d_hid[3])
            self.fc5 = nn.Linear(d_hid[3], d_out)
            self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
            self.bn4 = nn.BatchNorm1d(d_hid[3]) if batch_norm else None

        self.use_bn = batch_norm
        self.n_layers = len(d_hid)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.use_bn else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.use_bn else x
        x = self.fc3(self.relu(x))
        if self.n_layers > 2:
            x = self.bn3(x) if self.use_bn else x
            x = self.fc4(self.relu(x))
            if self.n_layers > 3:
                x = self.bn4(x) if self.use_bn else x
                x = self.fc5(self.relu(x))

        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

    def get_code(self, batch_size, device, return_labels=True, labels=None):
        if labels is None:  # sample labels
            labels = torch.randint(self.n_labels, (batch_size, 1), device=device)
        code = torch.randn(batch_size, self.d_code, device=device)
        gen_one_hots = torch.zeros(batch_size, self.n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = torch.cat([code, gen_one_hots.to(torch.float32)], dim=1)
        # print(code.shape)
        if return_labels:
            return code, gen_one_hots
        else:
            return code


def make_dataset(n_classes, n_samples, n_rows, n_cols, noise_scale,
                 discrete=False, force_make_new=False, base_data_path='data/SYNTH2D/'):
  """
  @param n_classes:
  @param n_samples:
  @param n_rows:
  @param n_cols:
  @param noise_scale:
  @param discrete:
  @param force_make_new:
  @param base_data_path:

  @returns: the dataset, and a function to estimate the pdf
  """
  n_clusters = n_rows * n_cols
  assert n_clusters % n_classes == 0  # equal number of clusters per class
  assert n_samples % n_clusters == 0  # equal number  of samples per cluster
  assert (not discrete) or (noise_scale < 0.5)  # ensure no overlap in discrete case

  class_grid, center_grid, class_centers = create_centers(n_classes, n_clusters, n_rows, n_cols)
  print(class_grid)

  spec_str = specs_to_string(n_classes, n_samples, n_rows, n_cols, noise_scale, discrete)
  os.makedirs(os.path.join(base_data_path, spec_str), exist_ok=True)
  data_save_str = os.path.join(base_data_path, spec_str, 'samples.npz')
  if not force_make_new and os.path.exists(data_save_str):  # check if dataset already exists
    samples_mat = np.load(data_save_str)
    # print(samples_mat)
    data_samples = samples_mat['data']
    label_samples = samples_mat['labels']

  else:
    data_samples, label_samples = get_data_samples(center_grid, class_grid, n_rows, n_cols,
                                                   n_clusters, noise_scale, n_samples, discrete)
    np.savez(data_save_str, data=data_samples, labels=label_samples)

  if discrete:
    eval_func = get_discrete_in_out_test(class_centers, noise_scale)
  else:
    eval_func = get_mix_of_gaussian_pdf(class_centers, noise_scale)

  return data_samples, label_samples, eval_func, class_centers


def create_centers(n_classes, n_clusters, n_rows, n_cols):
  center_grid = np.stack([np.repeat(np.arange(n_rows)[:, None], n_cols, 1),
                          np.repeat(np.arange(n_cols)[None, :], n_rows, 0)], axis=2)

  # assign clusters to classes
  assert n_cols % n_classes == 0  # for now assume classes neatly fit into a row
  class_grid = np.zeros((n_rows, n_cols), dtype=np.int)
  class_centers = np.zeros((n_classes, n_clusters // n_classes, 2))
  centers_done_per_class = [0 for _ in range(n_classes)]
  next_class = 0
  for row_idx in range(n_rows):
    for col_idx in range(n_cols):
      class_grid[row_idx, col_idx] = next_class  # store class in grid
      class_centers[next_class, centers_done_per_class[next_class]] = center_grid[row_idx, col_idx]  # store center
      centers_done_per_class[next_class] += 1
      next_class = (next_class + 1) % n_classes

    next_class = (class_grid[row_idx, 0] + n_classes // 2) % n_classes
  return class_grid, center_grid, class_centers


def get_data_samples(center_grid, class_grid, n_rows, n_cols, n_clusters, noise_scale, n_samples, discrete):
  data_samples = []
  label_samples = []
  n_samples_per_cluster = n_samples // n_clusters
  for row_idx in range(n_rows):
    for col_idx in range(n_cols):
      data_samples.append(
        get_samples_from_center(center_grid[row_idx, col_idx], noise_scale, n_samples_per_cluster, discrete))
      label_samples.append(np.zeros((n_samples_per_cluster,), dtype=np.int) + class_grid[row_idx, col_idx])

  data_samples = np.concatenate(data_samples).astype(dtype=np.float32)
  label_samples = np.concatenate(label_samples)
  return data_samples, label_samples


def get_samples_from_center(center, noise_scale, n_samples, discrete):
  if discrete:
    square_sample = np.random.uniform(low=-noise_scale, high=noise_scale, size=(2*n_samples, 2))
    sample_norms = np.linalg.norm(square_sample, axis=1)
    sample = square_sample[sample_norms <= noise_scale][:n_samples]  # reject samples in corners
    sample += center
    if len(sample) < n_samples:
      print(len(sample), n_samples)
      more_samples = get_samples_from_center(center, noise_scale, n_samples - len(sample), discrete)
      sample = np.stack([sample, more_samples], axis=0)
  else:
    sample = np.random.normal(loc=center, scale=noise_scale, size=(n_samples, 2))
  return sample


def get_mix_of_gaussian_pdf(class_centers, noise_scale, get_avg_sample_prob=True, log_prob=True):
  print(class_centers.shape)
  n_classes, n_gaussians_per_class = class_centers.shape[:2]
  n_gaussians = n_classes * n_gaussians_per_class

  # def mix_of_gaussian_pdf(data, labels, return_per_class=False):
  #   # evaluate each class separately
  #   data_prob_by_class = []
  #   n_data_by_class = []
  #   for c_idx in range(n_classes):
  #     c_data = data[labels == c_idx]  # shape (n_c, 2)
  #     norms = np.linalg.norm(class_centers[c_idx].T[:, :, None] - c_data.T[:, None, :], axis=0)  # (c, n_c)
  #     # because we're using spherical gaussians, we can compute pdf as in 1D based on the norms
  #     gauss_probs = 1 / (np.sqrt(2 * np.pi) * noise_scale) * np.exp(-1/(2 * noise_scale**2) * norms**2)
  #
  #     prob_per_sample = np.sum(gauss_probs, axis=0) / (n_clusters_per_class * n_classes)
  #
  #     if log_prob:
  #       prob_per_sample = np.log(prob_per_sample)
  #
  #     if get_avg_sample_prob:
  #       prob_c = np.sum(prob_per_sample)
  #       n_data_by_class.append(len(c_data))
  #     else:
  #       prob_c = np.sum(prob_per_sample) if log_prob else np.prod(prob_per_sample)
  #
  #     data_prob_by_class.append(prob_c)
  #
  #   data_prob_by_class = np.asarray(data_prob_by_class)
  #   if get_avg_sample_prob:
  #     n_data_by_class = np.asarray(n_data_by_class)
  #     data_prob = np.sum(data_prob_by_class) / np.sum(n_data_by_class)
  #     data_prob_by_class = data_prob_by_class / n_data_by_class
  #   else:
  #     data_prob = np.sum(data_prob_by_class) if log_prob else np.prod(data_prob_by_class)
  #
  #   if return_per_class:
  #     return data_prob, data_prob_by_class
  #   else:
  #     return data_prob

  def mix_of_gaussian_pdf(data, labels):
    # evaluate each class separately
    data_prob_by_class = []
    for c_idx in range(n_classes):
      c_data = data[labels == c_idx]  # shape (n_c, 2)
      norms = np.linalg.norm(class_centers[c_idx].T[:, :, None] - c_data.T[:, None, :], axis=0)  # (c, n_c)
      # because we're using spherical gaussians, we can compute pdf as in 1D based on the norms
      gauss_probs = 1 / (np.sqrt(2 * np.pi) * noise_scale) * np.exp(-1/(2 * noise_scale**2) * norms**2) / n_gaussians
      prob_per_sample = np.sum(gauss_probs, axis=0)
      prob_per_sample = np.log(prob_per_sample)
      prob_c = np.sum(prob_per_sample)
      data_prob_by_class.append(prob_c)

    data_prob = np.sum(np.asarray(data_prob_by_class))
    return data_prob
  return mix_of_gaussian_pdf


def get_discrete_in_out_test(class_centers, noise_scale):
  n_classes, n_clusters_per_class = class_centers.shape[:2]

  def discrete_in_out_test(data, labels, return_per_class=False):
    # iterate through classes
    n_data_in = []
    n_data_out = []
    labels = labels.flatten()
    for c_idx in range(n_classes):
      print(data.shape, labels.shape, c_idx, (labels == c_idx).shape)
      c_data = data[labels == c_idx]  # shape (n_c, 2)
      # below: shape (2, c, 1) x (2, 1, n_c) -> (2, c, n_c) -> (c, n_c)
      norms = np.linalg.norm(class_centers[c_idx].T[:, :, None] - c_data.T[:, None, :], axis=0)
      # since each datapoint is within at most 1 centers' radius, we can sum the total number of times this is the case
      n_data_in_c = np.sum(norms <= noise_scale)
      assert n_data_in_c <= len(c_data)
      n_data_out_c = len(c_data) - n_data_in_c
      n_data_in.append(n_data_in_c)
      n_data_out.append(n_data_out_c)

    n_data_in = np.asarray(n_data_in)
    n_data_out = np.asarray(n_data_out)
    frac_in = np.sum(n_data_in) / len(data)

    if return_per_class:
      return frac_in, n_data_in, n_data_out
    else:
      return frac_in

  return discrete_in_out_test


def plot_data(data, labels, save_str, class_centers=None, subsample=None, center_frame=False, title=''):
  n_classes = int(np.max(labels)) + 1
  colors = ['r', 'b', 'g', 'y', 'orange', 'black', 'grey', 'cyan', 'magenta', 'brown']
  plt.figure()
  plt.title(title)
  if center_frame:
    plt.xlim(-0.5, n_classes - 0.5)
    plt.ylim(-0.5, n_classes - 0.5)

  for c_idx in range(n_classes):
    c_data = data[labels == c_idx]

    if subsample is not None:
      n_sub = int(np.floor(len(c_data) * subsample))
      c_data = c_data[np.random.permutation(len(c_data))][:n_sub]

    plt.scatter(c_data[:, 1], c_data[:, 0], label=c_idx, c=colors[c_idx], s=.1)

    if class_centers is not None:
      print(class_centers[c_idx, 0, :])
      plt.scatter(class_centers[c_idx, :, 1], class_centers[c_idx, :, 0], marker='x', c=colors[c_idx], s=50.)

  plt.xlabel('x')
  plt.ylabel('y')
  # plt.legend()
  plt.savefig(f'{save_str}.png')


def specs_to_string(n_classes, n_samples, n_rows, n_cols, noise_scale, discrete):
  prefix = 'disc' if discrete else 'norm'
  return f'{prefix}_k{n_classes}_n{n_samples}_row{n_rows}_col{n_cols}_noise{noise_scale}'


def string_to_specs(spec_string):
  specs_list = spec_string.split('_')

  assert specs_list[0] in {'disc', 'norm'}
  assert specs_list[1][0] == 'k'
  assert specs_list[2][0] == 'n'
  assert specs_list[3][:3] == 'row'
  assert specs_list[4][:3] == 'col'
  assert specs_list[5][:5] == 'noise'

  discrete = specs_list[0] == 'disc'
  n_classes = int(specs_list[1][1:])
  n_samples = int(specs_list[2][1:])
  n_rows = int(specs_list[3][3:])
  n_cols = int(specs_list[4][3:])
  noise_scale = float(specs_list[5][5:])
  return n_classes, n_samples, n_rows, n_cols, noise_scale, discrete


def make_data_from_specstring(spec_string):
  n_classes, n_samples, n_rows, n_cols, noise_scale, discrete = string_to_specs(spec_string)
  data_samples, label_samples, eval_func, class_centers = make_dataset(n_classes, n_samples, n_rows, n_cols,
                                                                       noise_scale, discrete)
  return data_samples, label_samples, eval_func, class_centers

def main():
    data_samples, label_samples, eval_func, class_centers = make_dataset(n_classes=5,
                                                                       n_samples=90000,
                                                                       n_rows=5,
                                                                       n_cols=5,
                                                                       noise_scale=0.2,
                                                                       discrete=False)

    print("This is data_samples shape: ", data_samples.shape)
    print("This is label_samples: ", label_samples)
    plot_data(data_samples, label_samples, 'synth_2d_data_plot', center_frame=True, title='')
    print("This is the NLL for 2d data: ", eval_func(data_samples, label_samples))

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)
    # device = 'cpu'


    """ generator training usine dp_mehp """
    # one-hot encoding of labels.
    X_train = data_samples
    n, input_dim = X_train.shape
    n_classes = max(label_samples)+1

    # one hot encode the labels
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = np.expand_dims(label_samples, 1)
    true_labels = onehot_encoder.fit_transform(y_train)

    """ 
    2. Train a generative model for producing synthetic data 
    """
    #mini_batch_size = 200
    #input_size = 10  # size of inputs to the generator
    #hidden_size_1 = 100
    #hidden_size_2 = 50
    #input_dim = 2  # dimension of data
    #output_size = input_dim + n_classes

    #model = FCCondGen(input_size=input_size, hidden_size_1=[hidden_size_1, hidden_size_2],
    #                         output_size=output_size, n_classes=n_classes, use_sigmoid=True, batch_norm=True).to(device)
    # d_code, d_hid, d_out, n_labels, use_sigmoid=True, batch_norm=True):

    """Setting the same parameters as in dp-merf case"""
    d_code=5
    d_hid="200,500,500,200"
    order=20
    epochs=100
    lr=0.01
    lr_decay=0.9
    batch_size=500
    # gamma=1 # fixed to 1. this only matters when two kernels are used
    method="combined"

    model = FCCondGen(d_code, d_hid, input_dim, n_classes, use_sigmoid=False, batch_norm=True).to(device)

    """ set the scale length """
    # sigma = heuristic_for_length_scale(X_train, input_dim)
    # print('we use a separate length scale on each coordinate of the data using the median heuristic')
    # sigma2 = sigma ** 2
    # sigma2 = np.mean(sigma**2)
    # sigma2 = 4.0
    # print("This is sigma2: ", sigma2)

    if method =='combined':
        sigma2_prod = 0.5
        rho_prod = find_rho(sigma2_prod, False)
        sigma2_sum = 10.0
        rho_sum = find_rho(sigma2_sum,False)
    elif method == 'prod':
        sigma2 = 0.5
        rho = find_rho(sigma2, False)  # With no separate_kernel_length
        print("This is rho: ", rho)
    elif method == 'sum':
        sigma2 = 4.0
        rho = find_rho(sigma2, False)  # With no separate_kernel_length
        print("This is rho: ", rho)
    else:
        print('we do not support the chosen method')

    #Compute the mean embedding sum kernel for real data (X_train)
#    print('computing mean embedding sum kernel of data')
#    data_sum_embedding = torch.zeros(input_dim*(order+1), n_classes, device=device)
#    for idx in range(n_classes):
#       print(idx)
#       idx_data = X_train[y_train.squeeze()==idx,:]
#       print("This is idx_data for each label: ", idx_data.shape)
#       phi_data = ME_with_HP_tab(torch.Tensor(idx_data).to(device), order, rho, device, n) 
#       data_sum_embedding[:,idx] = phi_data # this includes 1/n factor inside
#       del phi_data
#   print('done with computing mean embedding of data')
#   print("data sum kernel embedding: ", data_sum_embedding.shape)

    if method == "product":
        #Compute the mean embedding prod kernel real data (X_train)
        prod_kernel_embedding_dim=pow(order + 1, input_dim) #(C+1)**D
        data_embedding=torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)
        for idx in range(n_classes):
            idx_data = X_train[y_train.squeeze()==idx]
            idx_data=torch.from_numpy(idx_data)
            #print("this is idx_data shape: ", idx_data.shape)

            data_embedding[:, idx]=ME_with_HP_prod(idx_data, order, rho, device, batch_size, prod_kernel_embedding_dim)
        #print("data prod kernel embedding: ", data_embedding_prod_kernel.shape)

    elif method == "sum":
        #Compute the mean embedding sum kernel real data (X_train)
        # data_embedding_dim1 = torch.zeros(order+1, n_classes, device=device)
        # data_embedding_dim2 = torch.zeros(order+1, n_classes, device=device)
        # num_bumps = 5
        # per_bump = 3600
        data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)
        for idx in range(n_classes):
            idx_data = X_train[y_train.squeeze()==idx]

            # idx_data=torch.from_numpy(idx_data)
            # rho_array = np.zeros((5,2))
            # for bump in range(num_bumps):
                # data_bump = idx_data[bump*per_bump:(bump+1)*per_bump,:]

                # sigma = heuristic_for_length_scale(data_bump, input_dim)
                # sigma = np.median(data_bump,axis=0)
                # sigma2 = sigma ** 2
                # rho = find_rho(sigma2, True)  # With no separate_kernel_length
                # print("This is rho: ", rho)
                # rho_array[bump,:] = rho

            data_embedding[:, idx] = ME_with_HP(torch.Tensor(idx_data).to(device),
                                                     order, rho, device, n)

            # data_embedding_dim1[:, idx] = ME_with_HP(torch.Tensor(np.expand_dims(idx_data[:,0],axis=1)).to(device), order, rho, device, n)
            # data_embedding_dim2[:, idx] = ME_with_HP(torch.Tensor(np.expand_dims(idx_data[:,1],axis=1)).to(device), order, rho, device, n)


    elif method == "combined":
        #Compute the mean embedding for prod and sum kernel real data (X_train)
        prod_kernel_embedding_dim=pow(order + 1, input_dim) #(C+1)**D
        prod_data_embedding=torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)

        sum_data_embedding = torch.zeros(input_dim*(order+1), n_classes, device=device)
        for idx in range(n_classes):
            idx_data = X_train[y_train.squeeze()==idx]
            idx_data=torch.from_numpy(idx_data)

            prod_data_embedding[:, idx]=ME_with_HP_prod(idx_data, order, rho_prod, device, batch_size, prod_kernel_embedding_dim)
            sum_data_embedding[:, idx]=ME_with_HP(idx_data, order, rho_sum, device, n)

        #Once both emebddings are computed we concatenate them.
        # data_embedding=torch.cat((prod_data_embedding, sum_data_embedding), 0)
        # print("This is data_embedding shape for combined kernel: ", data_embedding.shape)

    
    else:
        print("Not the correct option.")
        sys.exit()

    """ Training """

    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay)
    num_iter = np.int(n / batch_size)
    lamb = 10.0

    print('start training the generator')

    for epoch in range(1,epochs + 1):
        # model.train()

        for i in range(num_iter):
          
            #Produce generated data.
            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code)  # batch_size by 2
            #print(gen_samples)
            _, gen_labels_numerical = torch.max(gen_labels, dim=1)

            if method == "product":

                #I don't need to sumsample dimension for prod kernel because there are only 2.
                synth_data_embedding=torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)

                for idx in range(n_classes):
                    #Compute mean embedding prod kernel synthetic data.
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    synth_data_embedding[:, idx]=ME_with_HP_prod(idx_synth_data, order, rho, device, batch_size, prod_kernel_embedding_dim)
              
            elif method == "sum":
                #Compute mean embedding sum kernel synthetic data.
                #per_bump = np.int(batch_size/5)
                # synth_data_embedding_dim1 = torch.zeros(order+1, n_classes, device=device)
                # synth_data_embedding_dim2 = torch.zeros(order + 1, n_classes, device=device)
                synth_data_embedding = torch.zeros(input_dim*(order + 1), n_classes, device=device)

                for idx in range(n_classes):
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    # for bump in range(num_bumps):
                    #     syn_data_bump = idx_synth_data[bump * per_bump:(bump + 1) * per_bump, :]
                        # data_embedding[:, bump, idx] = ME_with_HP_tab(torch.Tensor(data_bump).to(device), order, rho,
                        #                                               device, n)
                        # rho = rho_array[bump,:]
                    # synth_data_embedding_dim1[:, idx]=ME_with_HP(torch.unsqueeze(idx_synth_data[:,0],dim=1), order, rho, device, batch_size)
                    # synth_data_embedding_dim2[:, idx] = ME_with_HP(torch.unsqueeze(idx_synth_data[:, 1],dim=1), order, rho, device, batch_size)
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order,
                                                                   rho, device, batch_size)

            elif method == "combined":
                prod_synth_data_embedding=torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)

                sum_synth_data_embedding = torch.zeros(input_dim*(order+1), n_classes, device=device)

                for idx in range(n_classes):
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]

                    prod_synth_data_embedding[:, idx]=ME_with_HP_prod(idx_synth_data, order, rho_prod, device, batch_size, prod_kernel_embedding_dim)

                    sum_synth_data_embedding[:, idx]=ME_with_HP(idx_synth_data, order, rho_sum, device, batch_size)
                
                # synth_data_embedding=torch.cat((prod_synth_data_embedding, sum_synth_data_embedding), 0)

            #
            # if method == 'sum':
            #     loss = torch.sum((data_embedding_dim1 - synth_data_embedding_dim1)**2) + torch.sum((data_embedding_dim2 - synth_data_embedding_dim2)**2)
            # else:
            if method == 'combined':
                loss_prod = torch.sum((prod_data_embedding - prod_synth_data_embedding)**2)
                loss_sum  = torch.sum((sum_data_embedding - sum_synth_data_embedding)**2)

                loss = loss_prod + lamb*loss_sum
            else:
                loss = torch.sum((data_embedding - synth_data_embedding) ** 2)


            optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()
        # end for
    
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
        print('loss prod', loss_prod)
        print('loss sum times lamb', lamb*loss_sum)
        # print("This is the NLL for 2d synthetic data: ", eval_func(gen_samples, np.argmax(gen_labels.detach().cpu().numpy(), axis=1)))
        scheduler.step()

    """ Once we have a trained generator, we store synthetic data from it, plot it and compute NLL """
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=n,
                                                               n_labels=n_classes)
    #print("syn_data shape: ", syn_data.shape)
    syn_labels2=np.reshape(syn_labels, -1)
    #print("syn_labels: ", syn_labels2)

    if method == "product":
        filename='synth_2d_data_plot_prod_kernel'
    elif method == "sum":
        filename='synth_2d_data_plot_sum_kernel'
    elif method == "combined":
        filename='synth_2d_data_plot_prod_and_sum_kernel'
        
    plot_data(syn_data, syn_labels2, filename, center_frame=True, title='')
    print("This is the NLL for 2d synthetic data: ", eval_func(syn_data, syn_labels2))


if __name__ == '__main__':
    main()