import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
#import util
import random
import argparse
#import seaborn as sns
#sns.set()
# %matplotlib inline
from autodp import privacy_calibrator
from autodp import rdp_acct, rdp_bank
from all_aux_files_tab_data import find_rho_tab, ME_with_HP_tab, ME_with_HP_prod, heuristic_for_length_scale
from all_aux_files import find_rho
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings('ignore')
import os
from marginals_eval import gen_data_alpha_way_marginal_eval
from binarize_adult import binarize_data
from all_aux_files_tab_data import undersample

os.makedirs("tab_results", exist_ok=True)
os.makedirs("tab_results/adult", exist_ok=True)

# ############################## generative models to use ###############################
class Generative_Model_homogeneous_data(nn.Module):
  def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, out_fun):
    super(Generative_Model_homogeneous_data, self).__init__()

    self.input_size = input_size
    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2
    # self.hidden_size_3 = hidden_size_3
    self.output_size = output_size
    assert out_fun in ('lin', 'sigmoid', 'relu')

    self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
    self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
    self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
    self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
    # self.bn3 = torch.nn.BatchNorm1d(self.hidden_size_3)
    # self.fc4 = torch.nn.Linear(self.hidden_size_3, self.output_size)
    # self.bn4 = torch.nn.BatchNorm1d(self.output_size)
    if out_fun == 'sigmoid':
      self.out_fun = nn.Sigmoid()
    elif out_fun == 'relu':
      self.out_fun = nn.ReLU()
    else:
      self.out_fun = nn.Identity()

  def forward(self, x):
    hidden = self.fc1(x)
    relu = self.relu(self.bn1(hidden))
    output = self.fc2(relu)
    relu = self.relu(self.bn2(output))
    output = self.fc3(relu)
    # relu = self.relu(self.bn3(output))
    # output = self.fc4(relu)
    output = self.out_fun(output)
    # output = torch.round(output) # so that we make the output as categorical
    return output


class Generative_Model_heterogeneous_data(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, categorical_columns,
                 binary_columns):
        super(Generative_Model_heterogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.categorical_columns = categorical_columns
        self.binary_columns = binary_columns

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)

        output_binary = self.sigmoid(output[:, 0:len(self.binary_columns)])
        output_categorical = self.relu(output[:, len(self.binary_columns):])
        output_combined = torch.cat((output_binary, output_categorical), 1)
        # X = X[:, binary_columns + categorical_columns]

        return output_combined


# ####################################### beginning of main script #######################################
def rescale_dims(data):
  # assume min=0
  max_vals = np.max(data, axis=0)
  print('max vals:', max_vals)
  data = data / max_vals
  print('new max', np.max(data))
  return data, max_vals


def revert_scaling(data, base_scale):
  return data * base_scale


def CGF_func(sigma1, num_repeats):
    func_gaussian_1 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma1}, x)
    func = lambda x: num_repeats * func_gaussian_1(x)

    return func

def main():
  args, device = parse_arguments()
  seed = np.random.randint(0, 1000)
  print('seed: ', seed)

  # print('Hermite polynomial order: ', args.order_hermite)

  random.seed(seed)
  ############################### data loading ##################################
  # print("adult_cat dataset")  # this is heterogenous

  if args.dataset_name=='adult':
      data = np.load(f"../data/real/sdgym_{args.dataset}_adult.npy")
  else: # for census data, we take the first
      data = np.load(f"../data/real/sdgym_{args.dataset}_census.npy")
      # n_subsampled_datapoints = 20000 # to do a quick test, later remove this
      # data = data[np.random.permutation(data.shape[0])][:n_subsampled_datapoints]
      # np.save('census_small.npy', data)

  # print("Data shape: ", data.shape)


  if args.kernel == 'linear':
    data, unbin_mapping_info = binarize_data(data)
    # print('bin data shape', data.shape)
  else:
    unbin_mapping_info = None

  if args.norm_dims == 1:
    data, base_scale = rescale_dims(data)
  else:
    base_scale = None


  ###########################################################################
  # PREPARING GENERATOR

  X = data # without labels separated
  # X = data[0:10000,:]  # to test things faster
  n_classes = 1

  n_samples, input_dim = X.shape

  ######################################
  # MODEL

  # model specifics
  batch_size = np.int(np.round(args.batch_rate * n_samples))
  # print("minibatch: ", batch_size)

  input_size = 5
  hidden_size_1 = 400 * input_dim
  hidden_size_2 = 100 * input_dim

  # hidden_size_3 = 10 * input_dim

  output_size = input_dim
  out_fun = 'relu' if args.kernel == 'gaussian' else 'sigmoid'

  model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                            hidden_size_2=hidden_size_2,
                                            output_size=output_size,
                                            out_fun=out_fun).to(device)

  ####################### estimating length scale for each dimensoin ##################
  sigma2 = np.median(X, 0)
  sigma2[sigma2==0] = 0.9
  if args.dataset_name == 'census':
      hp = args.hyperparam
      if hp==0:
          # sigma2 = sigma2
          med = heuristic_for_length_scale(X, input_dim)
          print(f'heuristic suggests kernel length {med}')
          sigma2 = med ** 2
      else:
          sigma2 = hp*np.sqrt(sigma2)

  elif args.dataset_name == 'adult':
      if args.hyperparam == 1.0:
          if args.dataset=='simple':
              sigma2 = 0.2*np.sqrt(sigma2)
          else:
              sigma2 = sigma2
      else:
          med = heuristic_for_length_scale(X, input_dim)
          print(f'heuristic suggests kernel length {med}')
          sigma2 = med**2
  else:
      print(' we do not support this dataset ')


  rho = find_rho_tab(sigma2)
  order = args.order_hermite

  ########## data mean embedding ##########
  print("we are running private = " + str(args.is_private))
  if args.is_private:
      # print("private")
      delta = 1e-5
      if args.combined_kernel:
          if args.split_eps:
              privacy_param_sum = privacy_calibrator.gaussian_mech(args.split_eps_rate*args.epsilon, delta*0.5, k=1)
              print('Noise level sigma for sum kernel =', privacy_param_sum['sigma'])
              privacy_param_prod = privacy_calibrator.gaussian_mech((1-args.split_eps_rate)*args.epsilon, delta*0.5, k=args.epochs)
              print('Noise level sigma for prod kernel =', privacy_param_prod['sigma'])
              # When rate = 0.1 and epochs = 100, sigma for sum = 49 and sigma_for_prod = 55
              # When rate = 0.2 and epochs = 100, sigma for sum = 24 and sigma_for_prod = 62
              # When rate = 0.5 and epochs = 100, sigma for sum = 9.9 and sigma for prod = 99
              # When rate = 0.6 and epochs = 100, sigma_for_sum = 8.3 and sigma_for_prod = 124
              # When rate = 0.8 and epochs = 100, sigma_for_sum = 6.2 and sigma_for_prod = 248


          else: # we add noise to ME_prod in each epoch
              k = 1 + args.epochs  # because we add noise to the weights and means separately.
              privacy_param = privacy_calibrator.gaussian_mech(args.epsilon, delta, k=k)
              print(f'eps,delta = ({args.epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])
              # with epochs =100, sigma = 49
      else:
          k = 1
          privacy_param = privacy_calibrator.gaussian_mech(args.epsilon, delta, k=k)
          print(f'eps,delta = ({args.epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])


  """ compute the means """
  # this is for sum kernel only
  data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)

  if args.dataset_name == 'adult':
      chunk_size = 250
  else: # census
      chunk_size = 500

  emb_sum = 0
  for idx in range(n_samples // chunk_size + 1):
      data_chunk = data[idx * chunk_size:(idx + 1) * chunk_size].astype(np.float32)
      chunk_emb = ME_with_HP_tab(torch.Tensor(data_chunk).to(device), order, rho, device, n_samples)
      emb_sum += chunk_emb

  data_embedding[:,0] = emb_sum


  if args.is_private:

      if args.split_eps:
          std_sum = (2 * privacy_param_sum['sigma'] / n_samples)
          std_prod = (2 * privacy_param_prod['sigma'] / n_samples)
      else:
          std_sum = (2 * privacy_param['sigma'] / n_samples)
          std_prod = std_sum

      noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std_sum
      data_embedding = data_embedding + noise


  """ Training """
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  num_iter = np.int(n_samples / batch_size)
  print('number of iterations per epoch: ', num_iter)

  # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
  for epoch in range(args.epochs):  # loop over the dataset multiple times
      model.train()

      if args.combined_kernel:
          order_prod = args.order_hermite_prod
          dimensions_subsample = np.random.choice(input_dim, args.prod_dimension, replace=False)
          print("These are the dimensions sumsampled for the prod kernel: ", dimensions_subsample)
          rho_prod = rho[dimensions_subsample]  # same as the rho applied to the sum kernel, rho matched based on subsampled inputs
          prod_kernel_embedding_dim = pow(order_prod + 1, args.prod_dimension)  # (C+1)**D_prod
          data_embedding_prod_kernel = torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)

          print("Computing prod kernel mean embedding given a set of subsampled input dimensions at epoch {}".format(epoch))

          for idx in range(n_samples // chunk_size + 1):
              idx_real_data = data[idx * chunk_size:(idx + 1) * chunk_size].astype(np.float32)
              phi_data_prod_kernel = ME_with_HP_prod(torch.Tensor(idx_real_data[:, dimensions_subsample]).to(device), order_prod, rho_prod,
                                                    device, n_samples, prod_kernel_embedding_dim)
              data_embedding_prod_kernel[:, 0] += phi_data_prod_kernel # because n_classes is 1


          if args.is_private:
              # print('we add noise to the mean embedding prod kernel as is_private is set to True.')
              # Draw noise for the prod kernel mean emebdding as many times as epochs.
              noise_prod = torch.randn(data_embedding_prod_kernel.shape[0], data_embedding_prod_kernel.shape[1],
                                       device=device) * std_prod
              data_embedding_prod_kernel = data_embedding_prod_kernel + noise_prod


      for i in range(num_iter):

          feature_input = torch.randn((batch_size, input_size)).to(device)
          input_to_model = feature_input

          """ (2) produce data """
          outputs = model(input_to_model)

          """ (3) compute synthetic data's mean embedding """
          syn_data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)
          if args.combined_kernel:
              prod_kernel_embedding_dim = pow(order_prod + 1, args.prod_dimension)  # (C+1)**D_prod
              synth_data_embedding_prod_kernel = torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)

          for idx in range(n_classes):
              idx_syn_data = outputs
              phi_syn_data = ME_with_HP_tab(idx_syn_data, order, rho, device, batch_size)
              syn_data_embedding[:, idx] = phi_syn_data  # this includes 1/n factor inside
              if args.combined_kernel:
                synth_data_embedding_prod_kernel[:, idx] = ME_with_HP_prod(idx_syn_data[:, dimensions_subsample],
                                                                         order_prod, rho_prod, device, batch_size,
                                                                         prod_kernel_embedding_dim)

          if args.combined_kernel:
              loss_prod = torch.sum((data_embedding_prod_kernel - synth_data_embedding_prod_kernel)**2)
              loss_sum  = torch.sum((data_embedding - syn_data_embedding)**2)
              loss = loss_sum + args.gamma*loss_prod
              # loss = loss_prod
          else:
            loss = torch.sum((data_embedding - syn_data_embedding)**2)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
      print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
      if args.combined_kernel:
          print('loss_sum :', loss_sum)
          print('loss_prod :', loss_prod)
          print('loss_prod * gamma :', loss_prod * args.gamma)

          print(torch.norm(synth_data_embedding_prod_kernel))
          print(torch.norm(data_embedding_prod_kernel))

      if (epoch == args.epochs-1): # at the end we compute 3-way and 4-way marginals
          
          print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
          
          """ draw final data samples """
          if args.dataset_name == 'census':
              chunk_size = 2000
              generated_data = np.zeros(((n_samples//chunk_size)*chunk_size, input_dim))

              # n_samples = 199523
              # generated_data.shape = 198000, 41
              for idx in range(n_samples // chunk_size):
                  #print('%d of generating samples out of %d' %(idx, n_samples // chunk_size))
                  feature_input = torch.randn((chunk_size, input_size)).to(device)
                  input_to_model = feature_input
                  outputs = model(input_to_model)
                  samp_input_features = outputs
                  generated_data[idx * chunk_size:(idx + 1) * chunk_size,:] = samp_input_features.cpu().detach().numpy()

              generated_input_features_final = generated_data

          else:
              feature_input = torch.randn((n_samples, input_size)).to(device)
              input_to_model = feature_input
              outputs = model(input_to_model)

              samp_input_features = outputs

              generated_input_features_final = samp_input_features.cpu().detach().numpy()

          ##################################################################################################################

          if args.norm_dims == 1:
            generated_input_features_final = revert_scaling(generated_input_features_final, base_scale)

          # run marginals test
          if args.dataset_name == 'census':
              save_file = f"census_{args.dataset}_gen_eps_{args.epsilon}_{args.kernel}_kernel_" \
                          f"batch_rate_{args.batch_rate}_hp_{args.order_hermite}.npy"
              if args.save_data:
                  # save generated samples
                  path_gen_data = f"../data/generated/census"
                  os.makedirs(path_gen_data, exist_ok=True)
                  data_save_path = os.path.join(path_gen_data, save_file)
                  np.save(data_save_path, generated_input_features_final)
                  # print(f"Generated data saved to {path_gen_data}")
              else:
                  data_save_path = save_file

              real_data = f'../data/real/sdgym_{args.dataset}_census.npy'
              # real_data = f'census_small.npy'
              alpha = 2
              # then subsample datapoints, because this dataset is huge
              avg_tv3 = gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                           real_data_path=real_data,
                                           discretize=True,
                                           alpha=alpha,
                                           verbose=True,
                                           unbinarize=args.kernel == 'linear',
                                           unbin_mapping_info=unbin_mapping_info,
                                           # n_subsampled_datapoints=10000,
                                           gen_data_direct=generated_input_features_final)

              alpha = 3
              avg_tv4 = gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                                 real_data_path=real_data,
                                                 discretize=True,
                                                 alpha=alpha,
                                                 verbose=True,
                                                 unbinarize=args.kernel == 'linear',
                                                 unbin_mapping_info=unbin_mapping_info,
                                                 # n_subsampled_datapoints=1000,
                                                 gen_data_direct=generated_input_features_final)



          else:
              save_file = f"adult_{args.dataset}_gen_eps_{args.epsilon}_{args.kernel}_kernel_" \
                          f"batch_rate_{args.batch_rate}_hp_{args.order_hermite}.npy"
              if args.save_data:
                  # save generated samples
                  path_gen_data = f"../data/generated/adult"
                  os.makedirs(path_gen_data, exist_ok=True)
                  data_save_path = os.path.join(path_gen_data, save_file)
                  np.save(data_save_path, generated_input_features_final)
                  print(f"Generated data saved to {path_gen_data}")
              else:
                  data_save_path = save_file

              real_data = f'../data/real/sdgym_{args.dataset}_adult.npy'
              alpha = 3
              # real_data = 'numpy_data/sdgym_bounded_adult.npy'
              avg_tv3 = gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                               real_data_path=real_data,
                                               discretize=True,
                                               alpha=alpha,
                                               verbose=True,
                                               unbinarize=args.kernel == 'linear',
                                               unbin_mapping_info=unbin_mapping_info,
                                               gen_data_direct=generated_input_features_final)

              alpha = 4
              # real_data = 'numpy_data/sdgym_bounded_adult.npy'
              avg_tv4 = gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                               real_data_path=real_data,
                                               discretize=True,
                                               alpha=alpha,
                                               verbose=True,
                                               unbinarize=args.kernel == 'linear',
                                               unbin_mapping_info=unbin_mapping_info,
                                               gen_data_direct=generated_input_features_final)

          filename = f"tab_results/param_search_{args.dataset_name}.csv"
          file = open(filename, "a+")
          file.write(f"{args.dataset},{args.epsilon},{args.batch_rate}, {args.order_hermite_prod},{args.prod_dimension},{args.gamma},{args.combined_kernel},{args.epochs},{avg_tv3:.3f},{avg_tv4:.3f},{args.split_eps_rate} \n")
          file.close()

###################################################################################################


def parse_arguments():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # device = 'cpu' # to avoid cuda memory issues
  print('device is ', device)

  args = argparse.ArgumentParser()

  args.add_argument('--combined-kernel', default=True, help='If true use both sum and product kernels. If false, use sum kernel only')
  args.add_argument('--order-hermite', type=int, default=100, help='')
  args.add_argument('--epochs', type=int, default=1)
  # args.add_argument("--batch-rate", type=float, default=0.01) # for adult data
  args.add_argument("--batch-rate", type=float, default=0.05)  # for census data
  args.add_argument("--lr", type=float, default=0.0001)

  args.add_argument("--hyperparam", type=float, default=0) # either 1.0 or 0
  
  args.add_argument('--is-private', default=True, help='produces a DP mean embedding of data')
  args.add_argument("--epsilon", type=float, default=1.0)
  args.add_argument("--dataset", type=str, default='simple', choices=['bounded', 'simple'])
  args.add_argument("--dataset_name", type=str, default='census', choices=['census', 'adult'])
  args.add_argument('--kernel', type=str, default='gaussian', choices=['gaussian', 'linear'])
  # args.add_argument("--data_type", default='generated')  # both, real, generated
  args.add_argument("--save-data", type=int, default=0, help='save data if 1')

  # these are about product kernel
  args.add_argument('--prod-dimension', type=int, default=5, help='select the number of dimensions for product kernel')
  args.add_argument('--order-hermite-prod', type=int, default=10, help='order of HP for approximating the product kernel')

  # this value weighs losses between sum and product kernels' MEs
  args.add_argument('--gamma', type=float, default=1.0, help='a weight to apply to the ME under the product kernel')

  # assign different epsilons to ME_prod and ME_sum
  args.add_argument('--split-eps', default=True,
                    help='we split epsilon into two parts according to split-eps-rate')
  args.add_argument('--split-eps-rate', type=float, default=0.5, help='a proportion of epsilon assigned to ME_sum and the rest is assigned to ME_prod')

  #
  # when this value is set to True, we subsample datapoints to compute the ME under the product kernel
  # in order to use the privacy amplification
  # args.add_argument('--is-subsample-data', default=True, help='we subsample datapoints to compute the ME under the product kernel')
  # args.add_argument("--data-subsample-prob", type=float, default=0.1, help='we will use only the subsampled data to compute ME under prod kernel')  # for census data
  #
  #args.add_argument("--d_hid", type=int, default=200)
  args.add_argument("--norm-dims", type=int, default=0, help='normalize dimensions to same range if 1')

  arguments = args.parse_args()
  print("arg", arguments)
  return arguments, device


if __name__ == '__main__':
  main()
