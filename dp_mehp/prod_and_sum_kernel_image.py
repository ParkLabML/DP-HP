import torch
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from all_aux_files import FCCondGen, ConvCondGen, find_rho, ME_with_HP, get_mnist_dataloaders
from all_aux_files import get_dataloaders, log_args, test_results_subsampling_rate
from all_aux_files import synthesize_data_with_uniform_labels, flatten_features, log_gen_data
from all_aux_files import heuristic_for_length_scale, plot_mnist_batch
from all_aux_files import ME_with_HP_prod
from all_aux_files_tab_data import ME_with_HP_tab
from collections import namedtuple
from privacy_analysis_subsampled_MEs import privacy_param_func
import faulthandler
import matplotlib
matplotlib.use('Agg')
from autodp import privacy_calibrator
faulthandler.enable()  # prints stacktrace in case of segmentation fault


train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                       'train_data', 'test_data',
                                                       'n_features', 'n_data', 'n_labels', 'eval_func'])


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=0, help='sets random seed')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits or fashion')
  parser.add_argument('--create-dataset', action='store_true', default=True, help='if true, make 60k synthetic code_balanced')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during training')
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--gen-batch-size', '-gbs', type=int, default=1000)
  parser.add_argument('--embed-batch-size', '-ebs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=10)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')
  
  # MODEL DEFINITION
  parser.add_argument('--model-name', type=str, default='CNN', help='you can use CNN of FC')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--n-channels', '-nc', type=str, default='16,8', help='specifies conv gen kernel sizes')
  parser.add_argument('--gen-spec', type=str, default='500,500', help='specifies hidden layers of generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='5,5', help='specifies conv gen kernel sizes')

  # ALTERNATE MODES
  parser.add_argument('--multi-release', action='store_true', default=False, help='make embedding each batch, if true')
  parser.add_argument('--report_intermediate', action='store_true', default=False, help='')
  parser.add_argument('--loss-type', type=str, default='MEHP', help='how to approx mmd')
  parser.add_argument('--method', type=str, default='sum_kernel', help='')
  parser.add_argument('--sampling-rate-synth', type=float, default=0.1,  help='')
  parser.add_argument('--skip-downstream-model', action='store_true', default=False, help='')
  parser.add_argument('--order-hermite-sum', type=int, default=100, help='')
  parser.add_argument('--order-hermite-prod', type=int, default=20, help='')
  parser.add_argument('--heuristic-sigma', action='store_true', default=False)
  parser.add_argument("--separate-kernel-length", action='store_true', default=False) # heuristic-sigma has to be "True", to enable separate-kernel-length
  parser.add_argument('--kernel-length-sum', type=float, default=0.001, help='')
  parser.add_argument('--kernel-length-prod', type=float, default=0.001, help='')
  parser.add_argument('--prod-dimension', type=int, default=2, help='select the number of dimensions for product kernel')
  parser.add_argument('--gamma', type=float, default=1.0, help='gamma for product kernel')

  parser.add_argument('--debug-data', type=str, default=None, choices=['flip', 'flip_binary', 'scramble_per_label'])
  # DP SPEC
  parser.add_argument('--is-private', action='store_true', default=False, help='produces a DP mean embedding of data')
  parser.add_argument('--epsilon', type=float, default=1., help='epsilon for sum kernel in (epsilon, delta)-DP')
  parser.add_argument('--delta', type=float, default=1e-5, help='delta for sum kernel  (epsilon, delta)-DP')
  parser.add_argument('--split', action='store_true', default=False, help='produces a DP mean embedding of data')
  parser.add_argument('--split-sum-ratio', type=float, default=0.5, help='privacy budget for sum kernel')

  parser.add_argument('--subsampled-data', action='store_true', default=False, help='subsampling data on both kernels')
  parser.add_argument('--subsampled-data-ratio', type=float, default=0.1, help='proportion of the dataset used on both kernels')
  parser.add_argument('--sigma-sum', type=float, default=4., help='')
  parser.add_argument('--sigma-prod', type=float, default=4., help='')

  ar = parser.parse_args()

  preprocess_args(ar)
  log_args(ar.log_dir, ar)
  return ar




def preprocess_args(ar):
    if ar.log_dir is None:
        assert ar.log_name is not None
        ar.log_dir = ar.base_log_dir + ar.log_name + '/'
    if not os.path.exists(ar.log_dir):
        os.makedirs(ar.log_dir)
        
    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)

    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)
        assert ar.data in {'digits', 'fashion'}


def get_full_data_embedding_sum_kernel(data_pkg, order, rho, embed_batch_size, device, data_key, separate_kernel_length,
                            debug_data):
  embedding_train_loader, _ = get_mnist_dataloaders(embed_batch_size, embed_batch_size,
                                                    use_cuda=device, dataset=data_key,
                                                    debug_data=debug_data)
  print("This is the embedding_train_loader: ", embedding_train_loader.batch_size)

  # summing at the end uses unnecessary memory - leaving previous version in in case of errors with this one
  data_embedding_sum_kernel = torch.zeros(data_pkg.n_features * (order + 1), data_pkg.n_labels, device=device)
  for batch_idx, (data, labels) in enumerate(embedding_train_loader):
    data, labels = flatten_features(data.to(device)), labels.to(device)
    #print("this is data.shape: ", data.shape)
    #print(batch_idx)
    for idx in range(data_pkg.n_labels):
      idx_data = data[labels == idx]
      if separate_kernel_length:
        phi_data_sum_kernel = ME_with_HP_tab(idx_data, order, rho, device, data_pkg.n_data)
      else:
        phi_data_sum_kernel = ME_with_HP(idx_data, order, rho, device, data_pkg.n_data)
      data_embedding_sum_kernel[:, idx] += phi_data_sum_kernel

  del embedding_train_loader
  return data_embedding_sum_kernel


def perturb_data_embedding(data_embedding, epsilon, delta, n_data, device):
  k = 1
  privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
  print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])
  # print('we add noise to the data mean embedding as the private flag is true')
  # std = (2 * privacy_param['sigma'] * np.sqrt(data_pkg.n_features) / data_pkg.n_data)
  std = (2 * privacy_param['sigma'] / n_data)
  noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std

  print(f'before perturbation, mean and variance of data mean embedding are '
        f'{torch.mean(data_embedding)} and {torch.std(data_embedding)} ')
  data_embedding = data_embedding + noise
  print(f'after perturbation, mean and variance of data mean embedding are '
        f'{torch.mean(data_embedding)} and {torch.std(data_embedding)} ')
  return data_embedding

def main():
    """Load settings"""
    ar = get_args()
    print(ar)
    torch.manual_seed(ar.seed)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
  
    """Load data"""
    data_pkg = get_dataloaders(ar.data, ar.batch_size, ar.test_batch_size, use_cuda=device,
                               normalize=False, synth_spec_string=None, test_split=None,
                               debug_data=ar.debug_data)
    print("this is data_pkg.n_labels: ", data_pkg.n_labels)
  
    if ar.debug_data is not None:
      plot_mat = np.zeros((100, 28, 28))
      for idx in range(10):
        plot_mat[10*idx:10*(idx+1), :] = data_pkg.train_data.data[data_pkg.train_data.targets == idx][:10, :].numpy()

      plot_mat /= np.max(plot_mat)
      plot_mnist_batch(plot_mat, 10, 10, ar.log_dir + f'data_example', denorm=False)

    """ Define a generator """
    if ar.model_name == 'FC':
        model = FCCondGen(ar.d_code, ar.gen_spec, data_pkg.n_features, data_pkg.n_labels,
                          use_sigmoid=True, batch_norm=True, use_clamp=False).to(device)
    elif ar.model_name == 'CNN':
        model = ConvCondGen(ar.d_code, ar.gen_spec, data_pkg.n_labels, ar.n_channels, ar.kernel_sizes,
                            use_sigmoid=True, batch_norm=True).to(device)

    """ set the scale length """
    # num_iter = np.int(data_pkg.n_data / ar.batch_size)

    if ar.heuristic_sigma:

        print('we use the median heuristic for length scale')
        sigma = heuristic_for_length_scale(data_pkg.train_loader, data_pkg.n_features,
                                           ar.batch_size, data_pkg.n_data, device)
        
        sigma2_sum = np.median(sigma**2)
        sigma2_prod = np.median(sigma**2)

    else:
        sigma2_sum = ar.kernel_length_sum
        sigma2_prod=ar.kernel_length_prod

    print('sigma2 for the sum kernel is', sigma2_sum)
    print('sigma2 for the prod kernel is', sigma2_prod)

    rho_sum = find_rho(sigma2_sum, ar.separate_kernel_length)
    rho_prod = find_rho(sigma2_prod, ar.separate_kernel_length)
    order_sum = ar.order_hermite_sum
    order_prod=ar.order_hermite_prod

    # ev_thr = 1e-6  # eigen value threshold, below this, we wont consider for approximation
    # order = find_order(rho, ev_thr)
    # or_thr = ar.order_hermite
    # if order>or_thr:
    #     order = or_thr
    #     print('chosen order is', order)
    if ar.subsampled_data:
      dataset_embedding_sum_kernel = None
      batch_size_for_MEs =int(ar.subsampled_data_ratio * data_pkg.n_data)
      if ar.is_private:
          sigma1 = ar.sigma_sum # sigma to use to privatize ME under sum kernel in every few learning steps
          sigma2 = ar.sigma_prod # sigma to use to privatize ME under sum kernel in every few learning steps
          final_epsilon = privacy_param_func(sigma1, sigma2, ar.delta, ar.epochs, batch_size_for_MEs, data_pkg.n_data)
          print("This is epsilon computed via sigma_sum and sigma_prod: ", final_epsilon)
          std_sum = (2 * sigma1 / batch_size_for_MEs)
          std_prod=(2 * sigma2 / batch_size_for_MEs)

    else:
      print('computing mean embedding of data')
      dataset_embedding_sum_kernel = get_full_data_embedding_sum_kernel(data_pkg, order_sum, rho_sum, ar.embed_batch_size, device,
                                                  ar.data, ar.separate_kernel_length,
                                                  ar.debug_data)
      print('done with computing mean embedding of data for sum kernel')

      if ar.is_private:
          if ar.split:
            epsilon_sum=ar.split_sum_ratio*ar.epsilon
            epsilon_prod=(1-ar.split_sum_ratio)*ar.epsilon
            delta_sum=ar.delta/2
            delta_prod=ar.delta/2
            dataset_embedding_sum_kernel = perturb_data_embedding(dataset_embedding_sum_kernel, epsilon_sum, delta_sum, data_pkg.n_data, device)
            print('we add noise to the mean embedding sum kernel as is_private is set to True.')
            privacy_param_prod = privacy_calibrator.gaussian_mech(epsilon_prod, delta_prod, k=ar.epochs) 
            std_prod = (2 * privacy_param_prod['sigma'] / data_pkg.n_data)
          else:
            if not ar.subsampled_data:
              """ MJ: this only makes sense if you're using both kernels, if not k has to be set to 1. """
              privacy_param = privacy_calibrator.gaussian_mech(ar.epsilon, ar.delta, k= 1 + ar.epochs) 
              std = (2 * privacy_param['sigma'] / data_pkg.n_data)
              #Draw noise for the sum kernel mean embedding 
              noise = torch.randn(dataset_embedding_sum_kernel.shape[0], dataset_embedding_sum_kernel.shape[1], device=device) * std
              dataset_embedding_sum_kernel = dataset_embedding_sum_kernel + noise
          

          
      else:
          print('we do not add noise to the mean embedding as is_private is set to False.')

    """ Training """
    optimizer = torch.optim.Adam(list(model.parameters()), lr=ar.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
    #score_mat = np.zeros(ar.epochs)

    """Subsample dimensions for product kernel"""
    np.random.seed(ar.seed)
    #dimensions_subsample=np.random.choice(data_pkg.n_features, ar.prod_dimension, replace=False)
    #print("These are the dimensions sumsampled for the prod kernel: ", dimensions_subsample)

    print('start training the generator')

    for epoch in range(1, ar.epochs + 1):
        model.train()

        """Subsample dimensions for product kernel"""
        #np.random.seed(ar.seed)
        dimensions_subsample=np.random.choice(data_pkg.n_features, ar.prod_dimension, replace=False)
        print("These are the dimensions sumsampled for the prod kernel: ", dimensions_subsample)
        prod_kernel_embedding_dim=pow(order_prod + 1, ar.prod_dimension) #(C+1)**D_prod
        data_embedding_prod_kernel=torch.zeros((prod_kernel_embedding_dim, data_pkg.n_labels), device=device)

        if  ar.subsampled_data:
          dataset_embedding_sum_kernel = torch.zeros(data_pkg.n_features * (order_sum + 1), data_pkg.n_labels, device=device)
          #At each epoch we subsample the proportion of data ar.subsampled_data_ratio
          num_batches=data_pkg.n_data/ar.batch_size #total datapoints divided by the batch size 
          batches_ratio=int(num_batches*ar.subsampled_data_ratio)
          random_batches=np.random.choice(int(num_batches), batches_ratio, replace=False)
          print('Those are the random batches: ', random_batches)

          #Compute sum kernel embedding data for the randomly selected batches
          for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
              #print("This is batch_idx: ", batch_idx)
              if batch_idx in random_batches:
                data, labels = flatten_features(data.to(device)), labels.to(device)
                #print("This is data.shape: ", data.shape)
                for idx in range(data_pkg.n_labels):
                  idx_data = data[labels == idx]
                  #Sum kernel
                  phi_data_sum_kernel = ME_with_HP(idx_data, order_sum, rho_sum, device, batch_size_for_MEs)
                  dataset_embedding_sum_kernel[:, idx] += phi_data_sum_kernel
                  #Product kernel
                  phi_data_prod_kernel =ME_with_HP_prod(idx_data[:, dimensions_subsample], order_prod, rho_prod, device,  batch_size_for_MEs, prod_kernel_embedding_dim)
                  data_embedding_prod_kernel[:, idx] += phi_data_prod_kernel

          if ar.is_private:
            #TO DO: Check if this is ok (is this subsampled mech with or withou replacement?)
          #  k=2*ar.epochs
          #  params = privacy_calibrator.gaussian_mech(ar.epsilon, ar.delta, prob=ar.subsampled_data_ratio, k=k)
            noise_prod = torch.randn(data_embedding_prod_kernel.shape[0], data_embedding_prod_kernel.shape[1], device=device) * std_prod
            data_embedding_prod_kernel= data_embedding_prod_kernel + noise_prod

            noise_sum = torch.randn(dataset_embedding_sum_kernel.shape[0], dataset_embedding_sum_kernel.shape[1], device=device) * std_sum
            dataset_embedding_sum_kernel= dataset_embedding_sum_kernel + noise_sum


        else:
          #We don't subsample data in order to compute the data mean embedding
          print("Computing prod kernel mean embedding at epoch {}".format(epoch))
          for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
            data, labels = flatten_features(data.to(device)), labels.to(device)
            for idx in range(data_pkg.n_labels):
              idx_real_data = data[labels == idx]
              #Computing mean embedding product kernel for real data samples
              phi_data_prod_kernel =ME_with_HP_prod(idx_real_data[:, dimensions_subsample], order_prod, rho_prod, device, data_pkg.n_data, prod_kernel_embedding_dim)
              #print("This is phi_data_prod_kernel shape: ", phi_data_prod_kernel.shape)
              data_embedding_prod_kernel[:, idx] += phi_data_prod_kernel

          if ar.is_private:
            print('we add noise to the mean embedding prod kernel as is_private is set to True.')
            #Draw noise for the prod kernel mean emebdding as many times as epochs.
            if ar.split:
              noise_prod = torch.randn(data_embedding_prod_kernel.shape[0], data_embedding_prod_kernel.shape[1], device=device) * std_prod
            else:  
              noise_prod = torch.randn(data_embedding_prod_kernel.shape[0], data_embedding_prod_kernel.shape[1], device=device) * std

            data_embedding_prod_kernel= data_embedding_prod_kernel + noise_prod


        
        for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
            #print("This is batch_idx during training the generator: ", batch_idx)
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)
            #print("This is labels real data shape: ", labels.shape)

            gen_code, gen_labels = model.get_code(ar.batch_size, device)
            gen_samples = model(gen_code)  # batch_size by 784

            if not ar.multi_release:
                synth_data_embedding_sum_kernel = torch.zeros((data_pkg.n_features * (order_sum+1), data_pkg.n_labels), device=device)
                prod_kernel_embedding_dim=pow(order_prod + 1, ar.prod_dimension) #(C+1)**D_prod
                synth_data_embedding_prod_kernel=torch.zeros((prod_kernel_embedding_dim, data_pkg.n_labels), device=device)
                #data_embedding_prod_kernel=torch.zeros((prod_kernel_embedding_dim, data_pkg.n_labels), device=device)

                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(data_pkg.n_labels):
                    idx_synth_data = gen_samples[gen_labels_numerical == idx] #Shape is [num_datapoints, dimensions=784 ]
                    #idx_real_data=data[labels == idx] #Real data for computing the prod kernel mean embedding at each minibatch. Shape is [num_datapoints, dimensions=784 ]
                    #print("The idx_synth_data shape: ", idx_synth_data.shape)
                    #print("The idx_real_data shape: ", idx_real_data.shape)

                    #Computing mean embedding sum kernel for synthethic data.
                    synth_data_embedding_sum_kernel[:, idx] = ME_with_HP(idx_synth_data, order_sum, rho_sum, device, ar.batch_size)
                    #print("Synthetic data sum kernel mean embedding shape: ", synth_data_embedding_sum_kernel.shape)
                    #Computing mean embedding product kernel for synthetic data. 

                    synth_data_embedding_prod_kernel[:, idx]=ME_with_HP_prod(idx_synth_data[:, dimensions_subsample], order_prod, rho_prod, device, ar.batch_size, prod_kernel_embedding_dim) 
                    #Computing mean embedding product kernel for real data samples
                    #data_embedding_prod_kernel[:, idx]=ME_with_HP_prod(idx_real_data[:, dimensions_subsample], order_prod, rho_prod, device, ar.batch_size, prod_kernel_embedding_dim)


                loss_prod = torch.sum((data_embedding_prod_kernel - synth_data_embedding_prod_kernel)**2)
                loss_sum  = torch.sum((dataset_embedding_sum_kernel - synth_data_embedding_sum_kernel)**2)

                loss = loss_prod + ar.gamma*loss_sum
                

            else:
                pass

            optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()
        # end for
        
        print("This is prduct loss:" , loss_prod)
        print("this is sum loss: ", loss_sum)
        print("The gamma*sum_loss is : ", ar.gamma*loss_sum)
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), data_pkg.n_data, loss.item()))


        print(torch.norm(synth_data_embedding_prod_kernel))
        print(torch.norm(data_embedding_prod_kernel))
        
        
        log_gen_data(model, device, epoch, data_pkg.n_labels, ar.log_dir) #plot generated samples.
        scheduler.step()

    """ Once we have a trained generator, we store synthetic data from it and test them on logistic regression """
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=ar.batch_size,
                                                               n_data=data_pkg.n_data,
                                                               n_labels=data_pkg.n_labels)

    dir_syn_data = ar.log_dir + ar.data + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
#    final_score = test_gen_data(ar.log_name + '/' +  ar.data, ar.data, subsample=ar.sampling_rate_synth, custom_keys='logistic_reg')
#    data_tuple = datasets_colletion_def(syn_data, syn_labels,
#                                        data_pkg.train_data.data, data_pkg.train_data.targets,
#                                        data_pkg.test_data.data, data_pkg.test_data.targets)
    test_results_subsampling_rate(ar.data, ar.log_name + '/' + ar.data, ar.log_dir, ar.skip_downstream_model, ar.sampling_rate_synth)
    
  

    
if __name__ == '__main__':
  main()
