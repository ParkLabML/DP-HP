import numpy as np
import os
from binarize_adult import un_binarize_data


def discretize_mat(mat, domain):
  _, n_features = mat.shape
  mat = np.round(mat)
  for feat in range(n_features):
    mat[:, feat] = np.clip(mat[:, feat], a_min=domain[feat, 0], a_max=domain[feat, 1])
  return mat


def get_domain(mat):
  return np.stack([np.min(mat, axis=0), np.max(mat, axis=0)], axis=1)


def gen_data_alpha_way_marginal_eval(gen_data_path, real_data_path, alpha, discretize, verbose=True,
                                     unbinarize=False, unbin_mapping_info=None,
                                     n_subsampled_datapoints=None,
                                     gen_data_direct=None):

  gen_data = np.load(gen_data_path) if gen_data_direct is None else gen_data_direct
  real_data = np.load(real_data_path)

  domain = np.stack([np.min(real_data, axis=0), np.max(real_data, axis=0)], axis=1)

  if n_subsampled_datapoints is not None:
    gen_data = gen_data[np.random.permutation(gen_data.shape[0])][:n_subsampled_datapoints]
    real_data = real_data[np.random.permutation(real_data.shape[0])][:n_subsampled_datapoints]

  if unbinarize:
    if discretize:
      bin_domain = np.stack([np.zeros(gen_data.shape[1]), np.ones(gen_data.shape[1])], axis=1)
      gen_data = discretize_mat(gen_data, bin_domain)
    assert unbin_mapping_info is not None
    gen_data = un_binarize_data(gen_data, unbin_mapping_info)
  elif discretize:
    gen_data = discretize_mat(gen_data, domain)

  tv_scores = alpha_way_marginal_tv_distances(gen_data, real_data, domain, alpha, verbose)
  avg_tv = np.mean(tv_scores)
  print(f'average {alpha}-way marginal tv score: {avg_tv}. (data:{gen_data_path})')

  return avg_tv


def alpha_way_marginal_tv_distances(gen_data, real_data, domain, alpha, verbose=True):
  n_samples, n_features = real_data.shape
  idx_tuples = get_alpha_indices(n_features, alpha)
  tv_scores = []
  print("length od the idx_tuples: ", len(idx_tuples))
  #print(idx_tuples)
  for count, tup in enumerate(idx_tuples):
    gen_submat = submat_joint(gen_data, tup, domain)
    real_submat = submat_joint(real_data, tup, domain)
    tv_scores.append(total_variation_distance(gen_submat, real_submat))
    if verbose and count % 500 == 499:
      print(f'marginals done: {count+1}')
  return np.asarray(tv_scores)


def total_variation_distance(p, q):
  return 0.5 * np.sum(np.abs(p - q))


def get_alpha_indices(n_features, alpha):
  last_idx_tuples = [(k,) for k in range(n_features + 1 - alpha)]
  for a_count in range(alpha):
    next_idx_tuples = []
    for tup in last_idx_tuples:
      next_idx_tuples.extend([tup + (k,) for k in range(tup[-1] + 1, n_features)])
    last_idx_tuples = next_idx_tuples
  return last_idx_tuples


def submat_joint(mat, idx_tuple, domain):
  sub_mat = np.stack([mat[:, k] for k in idx_tuple], axis=1)
  n_bins = [domain[k, 1] - domain[k, 0] for k in idx_tuple]
  joint = np.histogramdd(sub_mat, bins=n_bins)[0] / sub_mat.shape[0]
  return joint


def privbayes_scores(alpha, verbose):
  epsilons = ['1.0', '10.0']
  thetas = ['0.2', '1.0', '5.0']
  d_types = ['bounded', 'simple']

  for d_type in d_types:
    real_data_string = f'numpy_data/sdgym_{d_type}_adult.npy'
    for eps in epsilons:
      for theta in thetas:
        gen_data_string = f'gen_data/gen_sdgym_{d_type}_eps{eps}_theta{theta}.npy'
        gen_data_alpha_way_marginal_eval(gen_data_string, real_data_string, alpha, discretize=False, verbose=verbose)


def main():
  dtype = 'simple'
  # kernel = 'linear'
  kernel = 'gaussian'
  n_iter = 4000
  n_features = 2000
  dpmerf_dir = '../data/generated/rebuttal_exp/'
  dpmerf_file = f'adult_{dtype}_gen_eps_1.0_{kernel}_kernel_it_{n_iter}_features_{n_features}.npy'
  real_data = f'../data/real/sdgym_{dtype}_adult.npy'

  alpha = 3
  # real_data = 'numpy_data/sdgym_bounded_adult.npy'
  gen_data_alpha_way_marginal_eval(gen_data_path=os.path.join(dpmerf_dir, dpmerf_file),
                                   real_data_path=real_data,
                                   discretize=True,
                                   alpha=alpha,
                                   verbose=True)

  # 4-way bounded adult:
  # dpmerf: average 4-way marginal tv score:    3.238e-05
  # privbayes: average 4-way marginal tv score: 2.193e-05
  # 3-way bounded adult:
  # dpmerf: average 3-way marginal tv score:    2.809e-05
  # dpmerf: average 3-way marginal tv score:    2.518e-05. (data:eps_1.0_gaussian_kernel_it_20000_features_2000.npy)
  # privbayes: average 3-way marginal tv score: 1.744e-05

  # 3-way simple adult:
  # dpmerf: average 3-way marginal tv score:    1.710e-05
  # dpmerf: average 3-way marginal tv score:    1.603e-05. (data:eps_1.0_gaussian_kernel_it_8000_features_2000.npy)
  # dpmerf: average 3-way marginal tv score:    1.696e-05. (data:eps_1.0_gaussian_kernel_it_100000_features_2000.npy)
  # privbayes: average 3-way marginal tv score: 1.100e-05


def gen_then_disc_eval():

  for dtype in ['simple', 'bounded']:
    dpmerf_dir = '../data/generated/'
    dpmerf_file = f'gen_then_disc_{dtype}_adult.npy'
    real_data = f'../data/real/sdgym_{dtype}_adult.npy'

    alpha = 3
    gen_data_alpha_way_marginal_eval(gen_data_path=os.path.join(dpmerf_dir, dpmerf_file),
                                     real_data_path=real_data,
                                     discretize=True,
                                     alpha=alpha,
                                     verbose=True)

if __name__ == '__main__':
  # main()
  gen_then_disc_eval()
  # privbayes_scores(4, verbose=False)