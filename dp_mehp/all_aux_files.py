""" this contains all the relevant scripts, copied from the code_balanced folder"""

import torch as pt
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict, namedtuple, Iterable
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import xgboost
import sys

class NamedArray:
  def __init__(self, array, dim_names, idx_names):
    assert isinstance(array, np.ndarray) and isinstance(idx_names, dict)
    assert len(dim_names) == len(idx_names.keys()) and len(dim_names) == len(array.shape)
    for idx, name in enumerate(dim_names):
      assert len(idx_names[name]) == array.shape[idx], f'len({idx_names[name]}) != {array.shape[idx]} at idx={idx}'
      assert name in idx_names, f'{name} not in {idx_names}'
    self.array = array
    self.dim_names = dim_names  # list of dimension names in order
    self.idx_names = idx_names  # dict for the form dimension_name: [list of index names]

  def get(self, name_index_dict, keep_singleton_dims=False):
    """
    basically indexing by name for each dimension present in name_index_dict, it selects the given indices
    """
    for name in name_index_dict:
      assert name in self.dim_names
    ar = self.array
    for d_idx, dim in enumerate(self.dim_names):
      if dim in name_index_dict:
        names_to_get = name_index_dict[dim]
        # ids_to_get = [k for (k, name) in enumerate(self.idx_names[dim]) if name in names_to_get]
        ids_to_get = [self.idx_names[dim].index(name) for name in names_to_get]
        ar = np.take(ar, ids_to_get, axis=d_idx)

    if not keep_singleton_dims:
      ar = np.squeeze(ar)
    return ar

  def single_val_sub_array(self, dim_name, idx_name):
    assert dim_name in self.dim_names
    assert idx_name in self.idx_names[dim_name]

    sub_array = self.get({dim_name, [idx_name]})
    new_idx_names = self.idx_names.copy()
    new_idx_names[dim_name] = [idx_name]
    return NamedArray(sub_array, self.dim_names.copy(), new_idx_names)

  def merge(self, other, merge_dim):
    """
    merges another named array with this one:
    dimension names must be the same and in the same order
    in merge dimension: create union of index names (must be disjunct)
    in all other dimenions: create intersection of index names (must not be empty)
    """
    if isinstance(other, Iterable):
      m_arr = self
      for arr in other:
        m_arr = m_arr.merge(arr, merge_dim)
      return m_arr

    assert isinstance(other, NamedArray)
    assert merge_dim in self.dim_names
    assert all([n1 == n2 for n1, n2 in zip(self.dim_names, other.dim_names)])  # assert same dim names
    assert not [k for k in self.idx_names[merge_dim] if k in other.idx_names[merge_dim]]  # assert merge ids disjunct
    for dim in self.dim_names:
      if dim != merge_dim:
        assert any([k for k in self.idx_names[dim] if k in other.idx_names[dim]])  # assert intersection not empty

    self_dict = {}
    other_dict = {}
    merged_idx_names = {}
    # go through dims and construct index_dict for both self and other
    for d_idx, dim in enumerate(self.dim_names):
      if dim == merge_dim:
        self_dict[dim] = self.idx_names[dim]
        other_dict[dim] = other.idx_names[dim]
        merged_idx_names[dim] = self.idx_names[dim] + other.idx_names[dim]
      else:
        intersection = [k for k in self.idx_names[dim] if k in other.idx_names[dim]]
        self_dict[dim] = intersection
        other_dict[dim] = intersection
        merged_idx_names[dim] = intersection

    # then .get both sub-arrays and concatenate them
    self_sub_array = self.get(self_dict, keep_singleton_dims=True)
    other_sub_array = other.get(other_dict, keep_singleton_dims=True)
    print(self_sub_array.shape, other_sub_array.shape)
    merged_array = np.concatenate([self_sub_array, other_sub_array], axis=self.dim_names.index(merge_dim))

    # create new NamedArray instance and return it
    return NamedArray(merged_array, self.dim_names, merged_idx_names)


def find_rho(sigma2, kernel_separate):
  alpha = 1 / (2.0 * sigma2)
  rho = -1 / 2 / alpha + np.sqrt(1 / alpha ** 2 + 4) / 2
  rho_1 = -1 / 2 / alpha - np.sqrt(1 / alpha ** 2 + 4) / 2

  if kernel_separate:
      if (rho>1).any():
          print('some of the rho values are above 1. Mehler formula does not hold')
  else:
      
      if rho < 1:  # rho is always non-negative
          print('rho is less than 1. so we take this value.')
      elif rho > 1:
          print('rho is larger than 1. Mehler formula does not hold')
          if rho_1 > -1:  # rho_1 is always negative
              print('rho_1 is larger than -1. so we take this value.')
              rho = rho_1
          else:  # if rho_1 <-1,
              print('rho_1 is smaller than -1. Mehler formula does not hold')
              sys.exit('no rho values satisfy the Mehler formulas. We have to stop the run')

  return rho


def find_order(rho, eigen_val_threshold):
  k = 100
  eigen_vals = (1 - rho) * (rho ** np.arange(0, k + 1))
  idx_keep = eigen_vals > eigen_val_threshold
  keep_eigen_vals = eigen_vals[idx_keep]
  # print('keep_eigen_vals are ', keep_eigen_vals)
  order = len(keep_eigen_vals)
  # print('The number of orders for Hermite Polynomials is', order)
  return order


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
  # print("eigenvalues", eigen_vals)
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

  sum_val = torch.sum(phi_x, axis=0)
  phi_x = sum_val / n_training_data

  phi_x = phi_x / np.sqrt(input_dim)  # because we approximate k(x,x') = \sum_d k_d(x_d, x_d') / input_dim
  #print("This is the phi_x shape before reshaping it to (C+1)*D: ", phi_x) #this has shape: [imput_dim, (order+1)]

  phi_x = phi_x.view(-1)  # size: input_dim*(order+1)
  #print("This is the phi_x shape after reshaping it to (C+1)*D: ", phi_x)

  return phi_x

def ME_with_HP_prod(x, order, rho, device, n_training_data, dim_0_prod_kernel):
  n_data, input_dim = x.shape

  ### this is for debugging ###
  # with this code for 2-d data, the NLL is 3.7 x 10^5 after 50 epochs (Hermite order set to 10)
  #                              the NLL is 3.5 x 10^5 after 100 epochs (Hermite order set to 20)

  # per dimension, I will compute the features first
  #phi_x0, ev0 = feature_map_HP(order, x[:, 0], rho, device)
  #phi_x1, ev1 = feature_map_HP(order, x[:, 1], rho, device)
  #print("This is phi_x0: ", phi_x0)
  #print("This is phi_x1: ", phi_x1)

  # I will now compute the outer product between dimensions per datapoint , then take the average , then flatten the feature map
  #A = torch.einsum('ij, ik -> ijk', phi_x0, phi_x1)
  #A = torch.mean(A, axis=0)
  #out = A.view(-1)
 

  #test my code
  x_flattened = x.view(-1)
  x_flattened = x_flattened[:, None]
  phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_HP(order, x_flattened, rho, device)
  phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order + 1)
  #print("This is phi_x: ", phi_x)

  for dim in range(input_dim):
    if dim == 0:
      phi_0=phi_x[:, dim, :]
      #print("This is phi_0 dim: ", phi_0.shape)
    elif dim == 1:
      phi_1=phi_x[:, dim, :]
      B=torch.einsum('i...j, ik -> i...jk', phi_0, phi_1)
    else:
      phi_dim=phi_x[:, dim, :]
      B=torch.einsum('i...j, ik -> i...jk', B, phi_dim)
  
  mean_outer= torch.mean(B, axis=0)
  out = mean_outer.view(-1)

  return out

  ### Margarita's previous code ###
  # #print("this is input dimension of the data for prod kernel: ", input_dim)
  #
  # # reshape x, such that x is a long vector
  # x_flattened = x.view(-1)
  # x_flattened = x_flattened[:, None]
  # phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_HP(order, x_flattened, rho, device)
  # phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order + 1)
  # phi_x_prod_dims= phi_x.type(torch.float)
  # #phi_x_prod_dims=phi_x[:, dim_subsample, :]
  # #print("This is phi_x shape for ME_with_HP_prod: ", phi_x_prod_dims.shape)
  # outer_prod_per_datapoint=torch.zeros((dim_0_prod_kernel, n_data), device=device)
  #
  # for data in range(n_data):
  #   datapoint_phi_x=phi_x_prod_dims[data, :, :] #Select each datapoint in minibatch
  #   for dim in range(input_dim):
  #     if dim == 0:
  #       phi_0=datapoint_phi_x[dim, :]
  #     elif dim == 1:
  #       phi_1=datapoint_phi_x[dim, :]
  #       outer_prod=torch.outer(phi_0, phi_1)
  #     else:
  #       phi_i=datapoint_phi_x[dim, :]
  #       outer_prod=torch.einsum('i...j, k -> i...jk', outer_prod, phi_i)
  #
  #   outer_prod_per_datapoint[:, data]=outer_prod.view(-1)
  #
  # #print("After running the whole minibatch the outer_prod_phi_per_datapoint is: ", outer_prod_per_datapoint)
  # #print("The outer_prod_per_datapoint shape is: ", outer_prod_per_datapoint.shape )
  #
  # #Sum outer product over all datapoints in x and divide by number of datapoints in batch_size.
  # sum_outer_prod=torch.sum(outer_prod_per_datapoint, axis=1)
  # sum_outer_prod=sum_outer_prod / n_training_data
  # #print("This is the outer prod after summing over datapoints: ", sum_outer_prod.shape)
  #
  # #Multiply by gamma factor the prod kernel mean embedding
  # # sum_outer_prod=np.sqrt(gamma) * sum_outer_prod
  #
  # return sum_outer_prod


datasets_colletion_def = namedtuple('datasets_collection', ['x_gen', 'y_gen',
                                                            'x_real_train', 'y_real_train',
                                                            'x_real_test', 'y_real_test'])

def synthesize_data_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  if n_data % gen_batch_size != 0:
    assert n_data % 100 == 0
    gen_batch_size = n_data // 100
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():
    for idx in range(n_iterations):
      gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
      gen_samples = gen(gen_code)
      data_list.append(gen_samples)
  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


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


def subsample_data(x, y, frac, balance_classes=True):
  n_data = y.shape[0]
  n_classes = np.max(y) + 1
  new_n_data = int(n_data * frac)
  if not balance_classes:
    x, y = x[:new_n_data], y[:new_n_data]
  else:
    n_data_per_class = new_n_data // n_classes
    assert n_data_per_class * n_classes == new_n_data
    # print(f'starting label count {[sum(y == k) for k in range(n_classes)]}')
    # print('DEBUG: NCLASSES', n_classes, 'NDATA', n_data)
    rand_perm = np.random.permutation(n_data)
    x = x[rand_perm]
    y = y[rand_perm]
    # y_scalar = np.argmax(y, axis=1)

    data_ids = [[], [], [], [], [], [], [], [], [], []]
    n_full = 0
    for idx in range(n_data):
      l = y[idx]
      if len(data_ids[l]) < n_data_per_class:
        data_ids[l].append(idx)
        # print(l)
        if len(data_ids[l]) == n_data_per_class:
          n_full += 1
          if n_full == n_classes:
            break

    data_ids = np.asarray(data_ids)
    data_ids = np.reshape(data_ids, (new_n_data,))
    rand_perm = np.random.permutation(new_n_data)
    data_ids = data_ids[rand_perm]  # otherwise sorted by class
    x = x[data_ids]
    y = y[data_ids]

    # print(f'subsampled label count {[sum(y == k) for k in range(n_classes)]}')
  return x, y


def load_mnist_data(data_key, data_from_torch, base_dir='data/'):
  if not data_from_torch:
    if data_key == 'digits':
      d = np.load(
        os.path.join(base_dir, 'MNIST/numpy_dmnist.npz'))  # x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst

      return d['x_train'].reshape(60000, 784), d['y_train'], d['x_test'].reshape(10000, 784), d['y_test']
    elif data_key == 'fashion':
      d = np.load(os.path.join(base_dir, 'FashionMNIST/numpy_fmnist.npz'))
      return d['x_train'], d['y_train'], d['x_test'], d['y_test']
    else:
      raise ValueError
  else:
    from torchvision import datasets
    if data_key == 'digits':
      train_data = datasets.MNIST('data', train=True)
      test_data = datasets.MNIST('data', train=False)
    elif data_key == 'fashion':
      train_data = datasets.FashionMNIST('data', train=True)
      test_data = datasets.FashionMNIST('data', train=False)
    else:
      raise ValueError

    x_real_train, y_real_train = train_data.data.numpy(), train_data.targets.numpy()
    x_real_train = np.reshape(x_real_train, (-1, 784)) / 255

    x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()
    x_real_test = np.reshape(x_real_test, (-1, 784)) / 255
    return x_real_train, y_real_train, x_real_test, y_real_test

def prep_data(data_key, data_from_torch, data_path, shuffle_data, subsample, sub_balanced_labels):
  x_real_train, y_real_train, x_real_test, y_real_test = load_mnist_data(data_key, data_from_torch)
  gen_data = np.load(data_path)
  x_gen, y_gen = gen_data['data'], gen_data['labels']
  if len(y_gen.shape) == 2:  # remove onehot
    if y_gen.shape[1] == 1:
      y_gen = y_gen.ravel()
    elif y_gen.shape[1] == 10:
      y_gen = np.argmax(y_gen, axis=1)
    else:
      raise ValueError

  if shuffle_data:
    rand_perm = np.random.permutation(y_gen.shape[0])
    x_gen, y_gen = x_gen[rand_perm], y_gen[rand_perm]

  if subsample < 1.:
    x_gen, y_gen = subsample_data(x_gen, y_gen, subsample, sub_balanced_labels)
    x_real_train, y_real_train = subsample_data(x_real_train, y_real_train, subsample, sub_balanced_labels)

    # print(f'training on {subsample * 100.}% of the original syntetic dataset')

  # print(f'data ranges: [{np.min(x_real_test)}, {np.max(x_real_test)}], [{np.min(x_real_train)}, '
  #       f'{np.max(x_real_train)}], [{np.min(x_gen)}, {np.max(x_gen)}]')
  # print(f'label ranges: [{np.min(y_real_test)}, {np.max(y_real_test)}], [{np.min(y_real_train)}, '
  #       f'{np.max(y_real_train)}], [{np.min(y_gen)}, {np.max(y_gen)}]')

  return datasets_colletion_def(x_gen, y_gen, x_real_train, y_real_train, x_real_test, y_real_test)


def test_gen_data(data_log_name, data_key, data_base_dir='logs/gen/', log_results=True, data_path=None,
                  data_from_torch=False, shuffle_data=False, subsample=1., sub_balanced_labels=True,
                  custom_keys=None, skip_slow_models=False, only_slow_models=False,
                  skip_gen_to_real=False, compute_real_to_real=False, compute_real_to_gen=False,
                  print_conf_mat=False, norm_data=False):

  gen_data_dir = os.path.join(data_base_dir, data_log_name)
  log_save_dir = os.path.join(gen_data_dir, 'synth_eval/')
  if data_path is None:
    data_path = os.path.join(gen_data_dir, 'synthetic_mnist.npz')
  datasets_colletion = prep_data(data_key, data_from_torch, data_path, shuffle_data, subsample, sub_balanced_labels)
  mean_acc = test_passed_gen_data(data_key, data_log_name, datasets_colletion, log_save_dir, log_results,
                                  subsample, custom_keys, skip_slow_models, only_slow_models,
                                  skip_gen_to_real, compute_real_to_real, compute_real_to_gen,
                                  print_conf_mat, norm_data)
  return mean_acc

def prep_models(custom_keys, skip_slow_models, only_slow_models):
  assert not (skip_slow_models and only_slow_models)

  models = {'logistic_reg': linear_model.LogisticRegression,
            'random_forest': ensemble.RandomForestClassifier,
            'gaussian_nb': naive_bayes.GaussianNB,
            'bernoulli_nb': naive_bayes.BernoulliNB,
            'linear_svc': svm.LinearSVC,
            'decision_tree': tree.DecisionTreeClassifier,
            'lda': discriminant_analysis.LinearDiscriminantAnalysis,
            'adaboost': ensemble.AdaBoostClassifier,
            'mlp': neural_network.MLPClassifier,
            'bagging': ensemble.BaggingClassifier,
            'gbm': ensemble.GradientBoostingClassifier,
            'xgboost': xgboost.XGBClassifier}

  slow_models = {'bagging', 'gbm', 'xgboost'}

  model_specs = defaultdict(dict)
  #model_specs['gaussian_nb'] = {'var_smoothing': 1e-3}
  model_specs['logistic_reg'] = {'solver': 'lbfgs', 'max_iter': 50000, 'multi_class': 'auto'}
  model_specs['random_forest'] = {'n_estimators': 100, 'class_weight': 'balanced'}
  model_specs['linear_svc'] = {'max_iter': 10000, 'tol': 1e-8, 'loss': 'hinge'}
  model_specs['bernoulli_nb'] = {'binarize': 0.5}
  model_specs['lda'] = {'solver': 'eigen', 'n_components': 9, 'tol': 1e-8, 'shrinkage': 0.5}
  model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': 'gini', 'splitter': 'best',
                                  'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                                  'min_impurity_decrease': 0.0}
  #model_specs['adaboost'] = {'n_estimators': 100, 'algorithm': 'SAMME.R'}  # setting used in neurips2020 submission
  model_specs['adaboost'] = {'n_estimators': 1000, 'learning_rate': 0.7, 'algorithm': 'SAMME.R'}
  #best so far (not used for consistency with old results. change too small to warrant redoing everything)
  model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
  model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50}
  model_specs['xgboost'] = {'colsample_bytree': 0.1, 'objective': 'multi:softprob', 'n_estimators': 50}

  if custom_keys is not None:
    run_keys = custom_keys.split(',')
  elif skip_slow_models:
    run_keys = [k for k in models.keys() if k not in slow_models]
  elif only_slow_models:
    run_keys = [k for k in models.keys() if k in slow_models]
  else:
    run_keys = models.keys()

  return models, model_specs, run_keys




def normalize_data(x_train, x_test):
  mean = np.mean(x_train)
  sdev = np.std(x_train)
  x_train_normed = (x_train - mean) / sdev
  x_test_normed = (x_test - mean) / sdev
  assert not np.any(np.isnan(x_train_normed)) and not np.any(np.isnan(x_test_normed))

  return x_train_normed, x_test_normed


def model_test_run(model, x_tr, y_tr, x_ts, y_ts, norm_data, acc_str, f1_str):
  x_tr, x_ts = normalize_data(x_tr, x_ts) if norm_data else (x_tr, x_ts)
  model.fit(x_tr, y_tr)

  y_pred = model.predict(x_ts)
  acc = accuracy_score(y_pred, y_ts)
  f1 = f1_score(y_true=y_ts, y_pred=y_pred, average='macro')
  conf = confusion_matrix(y_true=y_ts, y_pred=y_pred)
  acc_str = acc_str + f' {acc}'
  f1_str = f1_str + f' {f1}'
  return acc, f1, conf, acc_str, f1_str


def test_passed_gen_data(data_key, data_log_name, datasets_colletion, log_save_dir, log_results=False,
                         subsample=1., custom_keys=None, skip_slow_models=False, only_slow_models=False,
                         skip_gen_to_real=False, compute_real_to_real=False, compute_real_to_gen=False,
                         print_conf_mat=False, norm_data=False):
  if data_log_name is not None:
    print(f'processing {data_log_name}')

  if log_results:
    os.makedirs(log_save_dir, exist_ok=True)

  models, model_specs, run_keys = prep_models(custom_keys, skip_slow_models, only_slow_models)
  print(model_specs)
  g_to_r_acc_summary = []
  dc = datasets_colletion
  for key in run_keys:
    # print(f'Model: {key}')
    a_str, f_str = 'acc:', 'f1:'

    if not skip_gen_to_real:
        if data_key == 'digits':
        #    if key == 'gaussian_nb':
        #        n_classes=10
        #        prior_class = 1/n_classes*np.ones(n_classes)
                
        #        list_acc=[]
        #        model_specs['gaussian_nb'] = {'var_smoothing': 1e-1}
        #        model1 = models[key](**model_specs[key])
        #        #print('model:', model1)
        #        g_to_r_acc1, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model1, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
        #        #print('accuracy of the 1st Gaussin_nb setting:', g_to_r_acc1) 
        #        list_acc.append(g_to_r_acc1)
                
        #        model_specs['gaussian_nb'] = {'var_smoothing': 1e-3}
        #        model2 = models[key](**model_specs[key])
        #        #print('model:', model2)
        #        g_to_r_acc2, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model2, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 2nd Gaussin_nb setting:', g_to_r_acc2) 
        #        list_acc.append(g_to_r_acc2)
                
        #        model_specs['gaussian_nb'] = {'var_smoothing': 1e-9}
        #        model3 = models[key](**model_specs[key])
        #        #print('model:', model3)
        #        g_to_r_acc3, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model3, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 3rd Gaussin_nb setting:', g_to_r_acc3) 
        #        list_acc.append(g_to_r_acc3)
                
        #        model_specs['gaussian_nb']={'var_smoothing': 0.2, 'priors' : prior_class}
        #        model4 = models[key](**model_specs[key])
        #        #print('model:', model4)
        #        g_to_r_acc4, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model4, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 4th Gaussin_nb setting:', g_to_r_acc4) 
        #        list_acc.append(g_to_r_acc4)
                
        #        model_specs['gaussian_nb']={'var_smoothing': 0.34, 'priors' : prior_class}
        #        model5 = models[key](**model_specs[key])
                #print('model:', model5)
        #        g_to_r_acc5, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model5, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 5th Gaussin_nb setting:', g_to_r_acc5) 
        #        list_acc.append(g_to_r_acc5)
                
        #        model_specs['gaussian_nb']={'var_smoothing': 0.12, 'priors' : prior_class}
        #        model6 = models[key](**model_specs[key])
                #print('model:', model6)
        #        g_to_r_acc6, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model6, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 6th Gaussin_nb setting:', g_to_r_acc6) 
        #        list_acc.append(g_to_r_acc6)
                
        #        model_specs['gaussian_nb']={'var_smoothing': 1e-2, 'priors' : prior_class}
        #        model7 = models[key](**model_specs[key])
                #print('model:', model7)
        #        g_to_r_acc7, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model7, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 7th Gaussin_nb setting:', g_to_r_acc7) 
        #        list_acc.append(g_to_r_acc7)
                
        #        model_specs['gaussian_nb']={'var_smoothing': 1e-1, 'priors' : prior_class}
        #        model8 = models[key](**model_specs[key])
                #print('model:', model8)
        #        g_to_r_acc8, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model8, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 8th Gaussin_nb setting:', g_to_r_acc8) 
        #        list_acc.append(g_to_r_acc8)
                
                
        #        g_to_r_acc=max(list_acc)
        #        print('This is the best accuacy for Gaussian_nb: ', g_to_r_acc)
        #        g_to_r_acc_summary.append(g_to_r_acc)
                
        #    elif key =='decision_tree':
        #        list_acc=[]
        #        list_criterion=['gini', 'entropy']
        #        list_min_samp_split=[2, 3, 4, 5, 8, 10, 12, 15, 20]
        #        for i in list_criterion:
        #            for j in list_min_samp_split:
        #                model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': i, 'min_samples_split': j}
        #                model1 = models[key](**model_specs[key])
                        #print('model:', model1)
        #                g_to_r_acc1, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model1, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                        #print('accuracy of the 1st Decision_tree setting:', g_to_r_acc1) 
        #                list_acc.append(g_to_r_acc1)
                
        #                model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': i ,'max_features':'log2', 'min_samples_split': j}
        #                model2 = models[key](**model_specs[key])
                        #print('model:', model2)
        #                g_to_r_acc2, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model2, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
        #                #print('accuracy of the 2nd Decision_tree setting:', g_to_r_acc2) 
        #                list_acc.append(g_to_r_acc2)
                
        #                model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': i ,'max_features':'auto', 'min_samples_split': j}
        #                model3 = models[key](**model_specs[key])
                        #print('model:', model3)
        #                g_to_r_acc3, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model3, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
        #                #print('accuracy of the 3rd Decision_tree setting:', g_to_r_acc3) 
        #                list_acc.append(g_to_r_acc3)
                
        #                model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': i,'max_features':'sqrt', 'min_samples_split': j}
        #                model4 = models[key](**model_specs[key])
                        #print('model:', model4)
        #                g_to_r_acc4, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model4, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                       # print('accuracy of the 4th Decision_tree setting:', g_to_r_acc4) 
        #                list_acc.append(g_to_r_acc4)
                        
        #        model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': 'gini','max_depth':100, 'max_leaf_nodes': 100}
        #        model5 = models[key](**model_specs[key])
                #print('model:', model5)
        #        g_to_r_acc5, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model5, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the final Decision_tree setting:', g_to_r_acc5) 
        #        list_acc.append(g_to_r_acc5)
                
        #        g_to_r_acc=max(list_acc)
        #        print('This is the best accuacy for Decision_tree: ', g_to_r_acc)
        #        g_to_r_acc_summary.append(g_to_r_acc)
                
        #    elif key =='linear_svc':
        #        list_acc=[]
                
        #        model_specs['linear_svc'] = {'max_iter': 10000, 'tol': 1e-8, 'loss': 'hinge'}
        #        model1 = models[key](**model_specs[key])
                #print('model:', model1)
        #        g_to_r_acc1, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model1, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 1st linear SVC setting:', g_to_r_acc1) 
         #       list_acc.append(g_to_r_acc1)
                
        #        model_specs['linear_svc'] = {'max_iter': 10000, 'tol': 1e-16, 'loss': 'hinge' ,  'multi_class' : 'crammer_singer', 'C' : 0.001}
        #        model2 = models[key](**model_specs[key])
                #print('model:', model2)
        #        g_to_r_acc2, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model2, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 1st linear SVC setting:', g_to_r_acc2) 
        #        list_acc.append(g_to_r_acc2)
                
        #        g_to_r_acc=max(list_acc)
        #        print('This is the best accuacy for linear_SVC: ', g_to_r_acc)
        #        g_to_r_acc_summary.append(g_to_r_acc)
                
        #    elif key =='bagging': 
        #        list_acc=[]
        #        model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
        #        model1 = models[key](**model_specs[key])
        #        #print('model:', model1)
        #        g_to_r_acc1, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model1, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
        #        #print('accuracy of the 1st Bagging setting:', g_to_r_acc1) 
        #        list_acc.append(g_to_r_acc1)
                
        #        model_specs['bagging'] = {'max_features': 40, 'n_estimators': 200, 'warm_start' : True} 
        #        model2 = models[key](**model_specs[key])
        #        #print('model:', model1)
        #        g_to_r_acc2, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model2, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 2nd Bagging setting:', g_to_r_acc1) 
        #        list_acc.append(g_to_r_acc2)
                
        #        g_to_r_acc=max(list_acc)
        #        print('This is the best accuacy for Bagging: ', g_to_r_acc)
        #        g_to_r_acc_summary.append(g_to_r_acc)
                
        #    elif key== 'adaboost':
        #       list_acc=[]
        #       model_specs['adaboost'] = {'n_estimators': 1000, 'learning_rate': 0.7, 'algorithm': 'SAMME.R'}  
        #       model1 = models[key](**model_specs[key])
               #print('model:', model1)
        #       g_to_r_acc1, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model1, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
               #print('accuracy of the 1st Adaboost setting:', g_to_r_acc1) 
        #       list_acc.append(g_to_r_acc1)
               
        #       model_specs['adaboost'] = {'n_estimators': 1000, 'learning_rate': 5.0, 'random_state': 0}  
        #       model2 = models[key](**model_specs[key])
               #print('model:', model1)
        #       g_to_r_acc2, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model2, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
               #print('accuracy of the 2nd Adaboost setting:', g_to_r_acc2) 
        #       list_acc.append(g_to_r_acc2)
               
        #       g_to_r_acc=max(list_acc)
        #       print('This is the best accuacy for Adaboost: ', g_to_r_acc)
        #       g_to_r_acc_summary.append(g_to_r_acc)
            
        #    elif key== 'gbm':
        #        list_acc=[] 
        #        model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50} 
        #        model1 = models[key](**model_specs[key])
                #print('model:', model1)
        #        g_to_r_acc1, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model1, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 1st gbm setting:', g_to_r_acc1) 
        #        list_acc.append(g_to_r_acc1)
                
        #        model_specs['gbm'] = {'subsample': 0.5, 'n_estimators': 100, 'learning_rate':0.05} 
        #        model2 = models[key](**model_specs[key])
                #print('model:', model1)
        #        g_to_r_acc2, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model2, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
                #print('accuracy of the 2nd gbm setting:', g_to_r_acc2) 
        #        list_acc.append(g_to_r_acc2)
                  
        #        g_to_r_acc=max(list_acc)
        #        print('This is the best accuacy for gbm: ', g_to_r_acc)
        #        g_to_r_acc_summary.append(g_to_r_acc) 
                
        #    else:    
        #        model = models[key](**model_specs[key])
        #        print('model:', model)
                
        #        g_to_r_acc, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model, dc.x_gen, dc.y_gen,
        #                                                                dc.x_real_test, dc.y_real_test,
        #                                                                norm_data, a_str + 'g2r', f_str + 'g2r')
        #        print('accuracy:', g_to_r_acc)
        #        g_to_r_acc_summary.append(g_to_r_acc)
        #
        # ORIGINAL SETTING
            model = models[key](**model_specs[key])
            print('model:', model)
                
            g_to_r_acc, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model, dc.x_gen, dc.y_gen,
                                                                        dc.x_real_test, dc.y_real_test,
                                                                        norm_data, a_str + 'g2r', f_str + 'g2r')
            print('accuracy:', g_to_r_acc)
            g_to_r_acc_summary.append(g_to_r_acc)    
                
        else:   
            #For fashion we apply the hyperparam set by default in prep_models
            model = models[key](**model_specs[key])
            print('model:', model)

            g_to_r_acc, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model, dc.x_gen, dc.y_gen,
                                                                        dc.x_real_test, dc.y_real_test,
                                                                        norm_data, a_str + 'g2r', f_str + 'g2r')

            print('accuracy:', g_to_r_acc)
            g_to_r_acc_summary.append(g_to_r_acc)
    else:
      g_to_r_acc, g_to_r_f1, g_to_r_conf = -1, -1, -np.ones((10, 10))

    if compute_real_to_real:
      model = models[key](**model_specs[key])
      base_acc, base_f1, base_conf, a_str, f_str = model_test_run(model,
                                                                  dc.x_real_train, dc.y_real_train,
                                                                  dc.x_real_test, dc.y_real_test,
                                                                  norm_data, a_str + 'r2r', f_str + 'r2r')
    else:
      base_acc, base_f1, base_conf = -1, -1, -np.ones((10, 10))

    if compute_real_to_gen:
      model = models[key](**model_specs[key])
      r_to_g_acc, r_to_g_f1, r_to_g_conv, a_str, f_str = model_test_run(model,
                                                                        dc.x_real_train, dc.y_real_train,
                                                                        dc.x_gen[:10000], dc.y_gen[:10000],
                                                                        norm_data, a_str + 'r2g', f_str + 'r2g')
    else:
      r_to_g_acc, r_to_g_f1, r_to_g_conv = -1, -1, -np.ones((10, 10))

    # print(a_str)
    # print(f_str)
    # if print_conf_mat:
    #   print('gen to real confusion matrix:')
    #   print(g_to_r_conf)

    if log_results:
      accs = np.asarray([base_acc, g_to_r_acc, r_to_g_acc])
      f1_scores = np.asarray([base_f1, g_to_r_f1, r_to_g_f1])
      conf_mats = np.stack([base_conf, g_to_r_conf, r_to_g_conv])
      file_name = f'sub{subsample}_{key}_log'
      np.savez(os.path.join(log_save_dir, file_name), accuracies=accs, f1_scores=f1_scores, conf_mats=conf_mats)

  # print('acc summary:')
  # for acc in g_to_r_acc_summary:
    # print(acc)
  mean_acc = np.mean(g_to_r_acc_summary)
  print(f'mean: {mean_acc}')
  return mean_acc


def test_results(data_key, log_name, log_dir, data_tuple, eval_func, skip_downstream_model):
  if data_key in {'digits', 'fashion'}:
    if not skip_downstream_model:
      final_score = test_gen_data(log_name, data_key, subsample=0.1, custom_keys='logistic_reg')
      log_final_score(log_dir, final_score)
#  elif data_key == '2d':
#    if not skip_downstream_model:
#      final_score = test_passed_gen_data(log_name, data_tuple, log_save_dir=None, log_results=False,
#                                         subsample=.1, custom_keys='mlp', compute_real_to_real=True)
#      log_final_score(log_dir, final_score)
#    eval_score = eval_func(data_tuple.x_gen, data_tuple.y_gen.flatten())
#    print(f'Score of evaluation function: {eval_score}')
#    with open(os.path.join(log_dir, 'eval_score'), 'w') as f:
#      f.writelines([f'{eval_score}'])

#    plot_data(data_tuple.x_real_train, data_tuple.y_real_train.flatten(), os.path.join(log_dir, 'plot_train'),
#              center_frame=True)
#    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen'))
#    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_sub0.2'), subsample=0.2)
#    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_centered'),
#              center_frame=True)

#    plot_data_1d(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_norms_hist'))
#  elif data_key == '1d':
#    plot_data_1d(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen'))
#    plot_data_1d(data_tuple.x_real_test, data_tuple.y_real_test.flatten(), os.path.join(log_dir, 'plot_data'))


def test_results_subsampling_rate(data_key, log_name, log_dir, skip_downstream_model, subsampling_rate):
  if data_key in {'digits', 'fashion'}:
    if not skip_downstream_model:
      final_score = test_gen_data(log_name, data_key, subsample=subsampling_rate, custom_keys='logistic_reg,random_forest,gaussian_nb,bernoulli_nb,linear_svc,decision_tree,lda,adaboost,mlp,bagging,gbm,xgboost')
      log_final_score(log_dir, final_score)

def log_args(log_dir, args):
  """ print and save all args """
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  with open(os.path.join(log_dir, 'args_log'), 'w') as f:
    lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    f.writelines(lines)
    for line in lines:
      print(line.rstrip())
  print('-------------------------------------------')

def log_final_score(log_dir, final_acc):
  """ print and save all args """
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(log_dir, 'final_score'), 'w') as f:
      lines = [f'acc: {final_acc}\n']
      f.writelines(lines)


def denormalize(mnist_mat):
  mnist_mean = 0.1307
  mnist_sdev = 0.3081
  return np.clip(mnist_mat * mnist_sdev + mnist_mean, a_min=0., a_max=1.)

def save_img(save_file, img):
  plt.imsave(save_file, img, cmap=cm.gray, vmin=0., vmax=1.)

def plot_mnist_batch(mnist_mat, n_rows, n_cols, save_path, denorm=True, save_raw=True):
  bs = mnist_mat.shape[0]
  n_to_fill = n_rows * n_cols - bs
  mnist_mat = np.reshape(mnist_mat, (bs, 28, 28))
  fill_mat = np.zeros((n_to_fill, 28, 28))
  mnist_mat = np.concatenate([mnist_mat, fill_mat])
  mnist_mat_as_list = [np.split(mnist_mat[n_rows*i:n_rows*(i+1)], n_rows) for i in range(n_cols)]
  mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)

  if denorm:
     mnist_mat_flat = denormalize(mnist_mat_flat)
  save_img(save_path + '.png', mnist_mat_flat)
  if save_raw:
    np.save(save_path + '_raw.npy', mnist_mat_flat)

def log_gen_data(gen, device, epoch, n_labels, log_dir):
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), n_labels)[:, None].to(device)
  gen_code, _ = gen.get_code(100, device, labels=ordered_labels)
  gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, n_labels, log_dir + f'samples_ep{epoch}', denorm=False)

class FCCondGen(nn.Module):
  def __init__(self, d_code, d_hid, d_out, n_labels, use_sigmoid=True, batch_norm=True, use_clamp=False):
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
    self.use_clamp = use_clamp

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

    if self.use_clamp:
      x = torch.clamp(x, min=-0.5, max=3.0)

    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code


class ConvCondGen(nn.Module):
  def __init__(self, d_code, d_hid, n_labels, nc_str, ks_str, use_sigmoid=True, batch_norm=True):
    super(ConvCondGen, self).__init__()
    self.nc = [int(k) for k in nc_str.split(',')] + [1]  # number of channels
    self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
    d_hid = [int(k) for k in d_hid.split(',')]
    assert len(self.nc) == 3 and len(self.ks) == 2
    self.hw = 7  # image height and width before upsampling
    self.reshape_size = self.nc[0]*self.hw**2
    self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
    self.fc2 = nn.Linear(d_hid[0], self.reshape_size)
    self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
    self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
    self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0]-1)//2)
    self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1]-1)//2)
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.bn1 is not None else x
    x = self.fc2(self.relu(x))
    x = self.bn2(x) if self.bn2 is not None else x
    # print(x.shape)
    x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
    x = self.upsamp(x)
    x = self.relu(self.conv1(x))
    x = self.upsamp(x)
    x = self.conv2(x)
    x = x.reshape(x.shape[0], -1)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code


class FCCondGen_2d(nn.Module):
  def __init__(self, d_code, d_hid, d_out, n_labels, use_sigmoid=False, spectral_norm=True):
    super(FCCondGen_2d, self).__init__()
    d_hid = [int(k) for k in d_hid.split(',')]
    assert len(d_hid) < 5

    if spectral_norm:
      self.fc1 = torch.nn.utils.spectral_norm(nn.Linear(d_code + n_labels, d_hid[0]))
      self.fc2 = torch.nn.utils.spectral_norm(nn.Linear(d_hid[0], d_hid[1]))

      if len(d_hid) == 2:
        self.fc3 = (nn.Linear(d_hid[1], d_out))
      elif len(d_hid) == 3:
        self.fc3 = torch.nn.utils.spectral_norm(nn.Linear(d_hid[1], d_hid[2]))
        self.fc4 = (nn.Linear(d_hid[2], d_out))
        # self.bn3 = nn.utils.spectral_norm(d_hid[2]) if spectral_norm else None
      elif len(d_hid) == 4:
        self.fc3 = torch.nn.utils.spectral_norm(nn.Linear(d_hid[1], d_hid[2]))
        self.fc4 = torch.nn.utils.spectral_norm(nn.Linear(d_hid[2], d_hid[3]))
        self.fc5 = nn.Linear(d_hid[3], d_out)
        # self.bn3 = nn.utils.spectral_norm(d_hid[2]) if spectral_norm else None
        # self.bn4 = nn.utils.spectral_norm(d_hid[3]) if spectral_norm else None
    else:
      print("probably it is a good idea to use the spectral norm?")


    # self.use_bn = spectral_norm
    self.n_layers = len(d_hid)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    # x = self.bn1(x) if self.use_bn else x
    x = self.fc2(self.relu(x))
    # x = self.bn2(x) if self.use_bn else x
    x = self.fc3(self.relu(x))
    if self.n_layers > 2:
      # x = self.bn3(x) if self.use_bn else x
      x = self.fc4(self.relu(x))
      if self.n_layers > 3:
        # x = self.bn4(x) if self.use_bn else x
        x = self.fc5(self.relu(x))

    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code


def meddistance(x, subsample=None, mean_on_fail=True):
  """
  Compute the median of pairwise distances (not distance squared) of points
  in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

  Parameters
  ----------
  x : n x d numpy array
  mean_on_fail: True/False. If True, use the mean when the median distance is 0.
      This can happen especially, when the data are discrete e.g., 0/1, and
      there are more slightly more 0 than 1. In this case, the m

  Return
  ------
  median distance
  """
  if subsample is None:
    d = dist_matrix(x, x)
    itri = np.tril_indices(d.shape[0], -1)
    tri = d[itri]
    med = np.median(tri)
    if med <= 0:
      # use the mean
      return np.mean(tri)
    return med

  else:
    assert subsample > 0
    rand_state = np.random.get_state()
    np.random.seed(9827)
    n = x.shape[0]
    ind = np.random.choice(n, min(subsample, n), replace=False)
    np.random.set_state(rand_state)
    # recursion just one
    return meddistance(x[ind, :], None, mean_on_fail)


def dist_matrix(x, y):
  """
  Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
  """
  sx = np.sum(x ** 2, 1)
  sy = np.sum(y ** 2, 1)
  d2 = sx[:, np.newaxis] - 2.0 * x.dot(y.T) + sy[np.newaxis, :]
  # to prevent numerical errors from taking sqrt of negative numbers
  d2[d2 < 0] = 0
  d = np.sqrt(d2)
  return d


def flatten_features(data):
  if len(data.shape) == 2:
    return data
  else:
    return pt.reshape(data, (data.shape[0], -1))


def flip_mnist_data(dataset, only_binary=True):
  data = dataset.data
  selections = np.zeros(data.shape[0], dtype=np.int)
  selections[:data.shape[0] // 2] = 1
  selections = pt.tensor(np.random.permutation(selections), dtype=pt.uint8)
  if only_binary:
    # print(selections.shape, data.shape, flipped_data.shape)
    dataset.data = pt.where(selections[:, None, None], pt.zeros_like(data) + 255, pt.zeros_like(data))
  else:
    flipped_data = 255 - data
    print(selections.shape, data.shape, flipped_data.shape, pt.max(data), pt.max(flipped_data))
    dataset.data = pt.where(selections[:, None, None], data, flipped_data)


def scramble_mnist_data_by_labels(dataset):
  data = dataset.data
  labels = dataset.targets
  oldshape = data.shape
  data_flat = pt.reshape(data, (oldshape[0], -1))
  # print(data.shape, labels.shape)
  print()
  new_data_list = []
  for label in range(10):  # shuffle each label separately
    # print(f'scrambling label {label}')
    l_data = data_flat[labels == label]
    # print(l_data.shape)
    n_data, n_dims = l_data.shape
    new_l_data = pt.zeros_like(l_data)
    for dim in range(l_data.shape[1]):
      new_l_data[:, dim] = l_data[:, dim][pt.randperm(n_data)]

    new_data_list.append(new_l_data)
  new_data_flat = pt.cat(new_data_list)

  dataset.targets = pt.cat([pt.zeros(new_data_list[k].shape[0]) + k for k in range(10)])
  dataset.data = pt.reshape(new_data_flat, oldshape)

train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                       'train_data', 'test_data',
                                                       'n_features', 'n_data', 'n_labels', 'eval_func'])


def get_dataloaders(dataset_key, batch_size, test_batch_size, use_cuda, normalize, synth_spec_string, test_split,
                    debug_data):
  if dataset_key in {'digits', 'fashion'}:
    train_loader, test_loader, trn_data, tst_data = get_mnist_dataloaders(batch_size, test_batch_size, use_cuda,
                                                                          dataset=dataset_key, normalize=normalize,
                                                                          return_datasets=True,
                                                                          debug_data=debug_data)
    n_features = 784
    n_data = 60_000
    n_labels = 10
    eval_func = None
  else:
    raise ValueError

  return train_data_tuple_def(train_loader, test_loader, trn_data, tst_data, n_features, n_data, n_labels, eval_func)


def get_mnist_dataloaders(batch_size, test_batch_size, use_cuda, normalize=False,
                          dataset='digits', data_dir='data', return_datasets=False,
                          debug_data=None):
  assert debug_data in (None, 'flip', 'flip_binary', 'scramble_per_label')

  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transforms_list = [transforms.ToTensor()]
  if dataset == 'digits':
    if normalize:
      mnist_mean = 0.1307
      mnist_sdev = 0.3081
      transforms_list.append(transforms.Normalize((mnist_mean,), (mnist_sdev,)))
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.MNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.MNIST(data_dir, train=False, transform=prep_transforms)
    if debug_data is not None and debug_data.startswith('flip'):
      assert not normalize
      print(pt.max(trn_data.data))
      flip_mnist_data(trn_data, only_binary=debug_data == 'flip_binary')
      flip_mnist_data(tst_data, only_binary=debug_data == 'flip_binary')

    if debug_data == 'scramble_per_label':
      scramble_mnist_data_by_labels(trn_data)
      scramble_mnist_data_by_labels(tst_data)

    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  elif dataset == 'fashion':
    assert not normalize
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.FashionMNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.FashionMNIST(data_dir, train=False, transform=prep_transforms)
    if debug_data is not None and debug_data.startswith('flip'):
      print(pt.max(trn_data.data))
      flip_mnist_data(trn_data, only_binary=debug_data == 'flip_binary')
      flip_mnist_data(tst_data, only_binary=debug_data == 'flip_binary')

    if debug_data == 'scramble_per_label':
      scramble_mnist_data_by_labels(trn_data)
      scramble_mnist_data_by_labels(tst_data)
    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  else:
    raise ValueError

  if return_datasets:
    return train_loader, test_loader, trn_data, tst_data
  else:
    return train_loader, test_loader


def heuristic_for_length_scale(train_loader, input_dim, batch_size, n_train_data, device):
    
    num_iter = np.int(n_train_data / batch_size)
    
    sigma_array = np.zeros((np.int(num_iter), input_dim))
    for batch_idx, (data, labels) in enumerate(train_loader):
    #     # print('batch idx', batch_idx)
         data, labels = data.to(device), labels.to(device)
         data = flatten_features(data)  # minibatch by feature_dim
         data_numpy = data.detach().cpu().numpy()
         for dim in np.arange(0, input_dim):
             med = meddistance(np.expand_dims(data_numpy[:, dim], axis=1))
             sigma_array[batch_idx, dim] = med


#    sigma_array = np.zeros(input_dim)
#    for i in np.arange(0, input_dim):
#        med = meddistance(np.expand_dims(X_train[:, i], 1), subsample=500)
#        sigma_array[i] = med
    return sigma_array
