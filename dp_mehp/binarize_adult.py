import numpy as np
from collections import OrderedDict


def binarize_data(data):
  n_samples, n_features = data.shape
  bin_features = []
  mapping_info = OrderedDict()
  for feat in range(n_features):
    bin_feat, map_tuple = binarize_feature(data[:, feat])
    bin_features.append(bin_feat)
    mapping_info[feat] = map_tuple
  bin_data = np.concatenate(bin_features, axis=1)
  return bin_data, mapping_info


def binarize_feature(feature):
  f_min, f_max = np.min(feature), np.max(feature)
  assert f_min != f_max
  n_bin_features = int(np.floor(np.log2(f_max - f_min) + 1))  # make log(domain) columns
  bin_features = np.zeros((feature.shape[0], n_bin_features))
  # mapping_base = np.arange(f_min, f_max+1)
  # mapping_bin = np.zeros((mapping_base.shape[0], n_bin_features))
  for idx in range(n_bin_features):
    bin_features[:, idx] = feature % 2
    # mapping_bin[:, idx] = mapping_base % 2
    feature = feature // 2
    # mapping_base = mapping_base // 2
  return bin_features, (n_bin_features, f_min)


def un_binarize_data(bin_data, mapping_info):
  bin_data_idx = 0
  unbin_features = []
  for feat_idx, (n_bin_features, f_min) in mapping_info.items():
    bin_data_chunk = bin_data[:, bin_data_idx:bin_data_idx+n_bin_features]
    bin_data_idx += n_bin_features
    unbin_features.append(un_binarize_feature(bin_data_chunk, f_min))

  return np.stack(unbin_features, axis=1)


def un_binarize_feature(bin_feature, f_min):
  n_samples, n_bin_features = bin_feature.shape
  feature = np.zeros((n_samples,)) + f_min
  for idx in range(n_bin_features):
    power = 2**idx
    feature += bin_feature[:, idx] * power
  return feature


def test():
  data = np.asarray([[0, 1, 4],
                     [1, 2, 4],
                     [2, 1, 4],
                     [3, 2, 4],
                     [4, 1, 4],
                     [5, 2, 4],
                     [6, 1, 4],
                     [7, 2, 6]
                     ])
  print(data)
  bin_data, mapping_info = binarize_data(data)
  print(bin_data)
  data2 = un_binarize_data(bin_data, mapping_info)
  print(data2)


if __name__ == '__main__':
  test()