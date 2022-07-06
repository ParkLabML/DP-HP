import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from aux import plot_mnist_batch
from aggregate_results import collect_results


DEFAULT_MODELS = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
                  'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
DEFAULT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray',
                  'tab:pink', 'limegreen', 'yellow']
DEFAULT_RATIOS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
DEFAULT_RATIO_KEYS = ['60k', '30k', '12k', '6k', '3k', '1.2k', '600', '300', '120', '60']

def dpcgan_plot():
  # loads = np.load('dp_cgan_synth_mnist_eps9.6.npz')
  loads = np.load('reference_dpcgan1_9.6.npz')
  # loads = np.load('ref_dpcgan_fashion5-eps9.6.npz')
  data, labels = loads['data'], loads['labels']

  print(np.sum(labels, axis=0))
  print(np.max(data), np.min(data))

  rand_perm = np.random.permutation(data.shape[0])
  data = data[rand_perm]
  labels = np.argmax(labels[rand_perm], axis=1)

  data_ids = [[], [], [], [], [], [], [], [], [], []]
  n_full = 0
  for idx in range(data.shape[0]):
    l = labels[idx]
    if len(data_ids[l]) < 10:
      data_ids[l].append(idx)
      # print(l)
      if len(data_ids[l]) == 10:
        n_full += 1
        if n_full == 10:
          break

  data_ids = np.asarray(data_ids)
  data_ids = np.reshape(data_ids, (100,))
  plot_mat = data[data_ids]
  plot_mnist_batch(plot_mat, 10, 10, 'dp_cgan_digit_plot', denorm=False, save_raw=False)


def dpgan_plot():
  data = np.load('dpgan_data.npy')

  rand_perm = np.random.permutation(data.shape[0])
  data = data[rand_perm] / 255.

  data = data[:100]
  print(np.max(data), np.min(data))
  plot_mnist_batch(data, 10, 10, 'dpgan_digit_plot', denorm=False, save_raw=False)


def plot_subsampling_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps',
            'dpmerf-nonprivate']
  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results_mean_subsampled.npy')
  print(mean_mat.shape)

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      # plt.xticks(DEFAULT_RATIOS[::-1], [str(k*100) for k in DEFAULT_RATIOS[::-1]])
      plt.xticks(DEFAULT_RATIOS[1:][::-1], [str(k * 100) for k in DEFAULT_RATIOS[::-1]])

      for s_idx, s in enumerate(setups):
        if s == 'real_data':
          continue
        # plt.plot(DEFAULT_RATIOS, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios
        plt.plot(DEFAULT_RATIOS[1:], mean_mat[d_idx, s_idx, 1:, e_idx], label=s)  # don_t show 1.0

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.hlines([0.4, 0.5, 0.6, 0.7, 0.8], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.legend()
      plt.savefig(f'plot_subsampling_{d}_{e}.png')


def mar19_plot_nonprivate_subsampling_performance():
  data_ids = ['d', 'f']
  setups = ['()', 'dpmerf-nonprivate', 'dpcgan-nonprivate', 'merf-AE-nonprivate', 'merf-DE-nonprivate']

  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results_mean_mar19_nonp.npy')

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(DEFAULT_RATIOS[::-1], [str(k * 100) for k in DEFAULT_RATIOS[::-1]])

      for s_idx, s in enumerate(setups):
        if s == '()':
          continue
        plt.plot(DEFAULT_RATIOS, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.hlines([0.4, 0.5, 0.6, 0.7, 0.8], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.legend()
      plt.savefig(f'mar19_nonp_{d}_{e}.png')


def plot_subsampling_logreg_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps']
  sub_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  model_idx = 1
  eval_metrics = ['accuracies', 'f1_scores']
  all_mat = np.load('results_full_subsampled.npy')
  mean_mat = np.mean(all_mat, axis=4)[:, :, :, model_idx, :]  # mean over runs, select logreg model

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'plot_{DEFAULT_MODELS[model_idx]}_{d}_{e}.png')


def plot_renorm_performance():
  data_ids = ['d']
  setups = ['real', 'base', 'renorm', 'clip']
  sub_ratios = [0.1, 0.01, 0.001]
  eval_metrics = ['accuracies']
  mean_mat = np.load('results_mean_renorm.npy')

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'renorm_plot_{d}_{e}.png')


def mar20_plot_sr_performance():
  data_ids = ['d', 'f']
  setups = ['()',
            'mar19_sr_rff1k_sig50', 'mar19_sr_rff10k_sig50', 'mar19_sr_rff100k_sig50',
            'mar19_sr_rff1k_sig5', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff100k_sig5',
            'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
            'mar19_sr_rff1k_sig0', 'mar19_sr_rff10k_sig0', 'mar19_sr_rff100k_sig0']

  sub_ratios = [0.1, 0.01]
  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results_mean_mar20_sr.npy')

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        if s == '()':
          continue
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s, color=DEFAULT_COLORS[s_idx])  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'mar20_sr_{d}_{e}.png')


def apr4_plot_subsampling_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'DP-CGAN eps=9.6', 'DP-MERF eps=1.3', 'DP-MERF eps=2.9', 'DP-MERF eps=9.6',
            'DP-MERF non-DP']

  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results/results_mean_subsampled.npy')
  print(mean_mat.shape)

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      # plt.xticks(DEFAULT_RATIOS[1:][::-1], [str(k * 100) for k in DEFAULT_RATIOS[::-1]])
      plt.xticks(DEFAULT_RATIOS[::-1], [str(k * 100) for k in DEFAULT_RATIOS[::-1]])

      for s_idx, s in enumerate(setups):
        if s == 'real_data':
          continue
        print(d_idx, e_idx, s_idx)
        # plt.plot(DEFAULT_RATIOS, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios
        # plt.plot(DEFAULT_RATIOS[1:], mean_mat[d_idx, s_idx, 1:, e_idx], label=s)  # don_t show 1.0
        plt.plot(DEFAULT_RATIOS, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # do show 1.0

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.hlines([0.45, 0.5, 0.55], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.ylim((0.4, 0.6))
      plt.legend()
      plt.savefig(f'apr4_plot_subsampling_{d}_{e}.png')


def spot_synth_mnist_mar19():
  rff = [1, 10, 100]
  sig = ['50', '5', '2.5', '0']
  dat = ['d', 'f']
  run = [0, 1, 2, 3, 4]
  for f in rff:
    for s in sig:
      for d in dat:
        for r in run:
          path = f'logs/gen/mar19_sr_rff{f}k_sig{s}_{d}{r}/synthetic_mnist.npz'
          if not os.path.isfile(path):
            print(f'{path} not found')


def plot_with_variance(x, y, color, label, alpha=0.1):
  """
  assume y is of shape (x_settings, runs to average)
  """
  means_y = np.mean(y, axis=1)
  sdevs_y = np.std(y, axis=1)
  # plt.plot(x, means_y, 'o-', label=label, color=color)
  # plt.fill_between(x, means_y-sdevs_y, means_y+sdevs_y, alpha=alpha, color=color)
  plot_with_variance_given_mean_and_sdev(x, means_y, sdevs_y, color, label, alpha)
  return means_y, sdevs_y


def plot_with_variance_given_mean_and_sdev(x, means_y, sdevs_y, color, label, alpha):
  plt.plot(x, means_y, 'o-', label=label, color=color)
  plt.fill_between(x, means_y - sdevs_y, means_y + sdevs_y, alpha=alpha, color=color)


def mar24_plot_selected_sr():
  data_ids = ['d', 'f']
  setups = ['mar19_sr_rff1k_sig50', 'mar19_sr_rff10k_sig50', 'mar19_sr_rff100k_sig50',
            'mar19_sr_rff1k_sig5', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff100k_sig5',
            'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
            'mar19_sr_rff1k_sig0', 'mar19_sr_rff10k_sig0', 'mar19_sr_rff100k_sig0']

  metric = 'accuracies'
  sub_ratios = [0.1, 0.01]

  _, _, sr_array, _, _, sr_d_array, sr_f_array = collect_results()

  for d_idx, d in enumerate(data_ids):

    plt.figure()
    plt.title(f'data: {d}, metric: {metric}')
    plt.xscale('log')
    plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

    for s_idx, s in enumerate(setups):
      # print(d, s, models)
      sub_mat = sr_array.get({'data_ids': [d], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})
      print(sub_mat.shape)
      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      plot_with_variance(sub_ratios, sub_mat, color=DEFAULT_COLORS[s_idx], label=s)
      # plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s, color=DEFAULT_COLORS[s_idx])  # plot over ratios

    plt.xlabel('% of data')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(f'mar24_sr_var_{d}_{metric}.png')


def mar25_plot_selected_sr():
  sr_d_setups = ['mar19_sr_rff10k_sig50', 'mar19_sr_rff1k_sig5', 'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff1k_sig0']
  sr_f_setups = ['mar19_sr_rff100k_sig50', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff10k_sig0']

  metric = 'accuracies'
  # metric = 'f1_scores'
  sub_ratios = [1.0, 0.1, 0.01]
  _, _, _, _, _, sr_d_array, sr_f_array = collect_results()

  # digit plot
  plt.figure(), plt.title(f'data: d, metric: {metric}'), plt.xscale('log')
  plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])
  for s_idx, s in enumerate(sr_d_setups):
    sub_mat = sr_d_array.get({'data_ids': ['d'], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})
    print('sr_d:', s)
    by_model = np.mean(sub_mat, axis=2)[0, :]  # average over runs, select 1.0 subsampling
    for v in by_model:
      print(v)
    sub_mat = np.mean(sub_mat, axis=1)  # average over models
    plot_with_variance(sub_ratios, sub_mat, color=DEFAULT_COLORS[s_idx], label=s)
  plt.xlabel('% of data'), plt.ylabel(metric), plt.legend()
  plt.savefig(f'plots/mar25_sr_var_d_{metric}.png')

  # digit plot
  plt.figure(), plt.title(f'data: f, metric: {metric}'), plt.xscale('log')
  plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])
  for s_idx, s in enumerate(sr_f_setups):
    sub_mat = sr_f_array.get({'data_ids': ['f'], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})
    print('sr_f:', s)
    by_model = np.mean(sub_mat, axis=2)[0, :]  # average over runs, select 1.0 subsampling
    for v in by_model:
      print(v)
    sub_mat = np.mean(sub_mat, axis=1)  # average over models
    plot_with_variance(sub_ratios, sub_mat, color=DEFAULT_COLORS[s_idx], label=s)

  plt.xlabel('% of data'), plt.ylabel(metric), plt.legend()
  plt.savefig(f'plots/mar25_sr_var_f_{metric}.png')


def mar25_plot_nonprivate():
  queried_setups = ['real_data', 'dpmerf-nonprivate', 'dpcgan-nonprivate', 'mar19_nonp_ae', 'mar19_nonp_de']
  metric = 'accuracies'
  # metric = 'f1_scores'
  data_ids = ['d', 'f']
  _, _, _, sb_np_array, _, _, _ = collect_results()

  # digit plot
  for d_id in data_ids:
    plt.figure(), plt.title(f'data: {d_id}, metric: {metric}'), plt.xscale('log')
    plt.xticks(DEFAULT_RATIOS[::-1], [str(k * 100) for k in DEFAULT_RATIOS[::-1]])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = sb_np_array.get({'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})
      print(f'sr_{d_id}:', s)
      by_model = np.mean(sub_mat, axis=2)[6, :]  # average over runs, select 1.0 subsampling
      for v in by_model:
        print(v)
      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=s)
    plt.xlabel('% of data'), plt.ylabel(metric), plt.legend()
    plt.savefig(f'plots/mar25_nonp_var_{d_id}_{metric}.png')


def apr4_plot_subsampling_performance_variance(plot_var=True):
  data_ids = ['MNIST', 'FashionMNIST']
  setups = ['real_data', 'DP-CGAN eps=9.6', 'DP-MERF eps=1.3', 'DP-MERF eps=2.9', 'DP-MERF eps=9.6',
            'DP-MERF non-DP']
  eval_metrics = ['accuracy', 'f1_score']
  mean_mat = np.load('results/results_full_mar25_subsampled.npy')
  print(mean_mat.shape)
  mean_mat = np.mean(mean_mat, axis=3)

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}')
      plt.xscale('log')
      plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])

      for s_idx, s in enumerate(setups):
        if s == 'real_data':
          continue
        if plot_var:
          print('yes')
          sub_mat = mean_mat[d_idx, s_idx, :, :, e_idx]
          plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=s)
        else:
          sub_mat = np.mean(mean_mat[d_idx, s_idx, :, :, e_idx], axis=1)
          plt.plot(DEFAULT_RATIOS, sub_mat, label=s, color=DEFAULT_COLORS[s_idx])  # do show 1.0

      plt.xlabel('# samples generated')
      plt.ylabel(e)
      if d == 'MNIST':
        plt.yticks([0.4, 0.45, 0.5, 0.55, 0.6])
        plt.hlines([0.45, 0.5, 0.55], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
        plt.ylim((0.4, 0.6))
      else:
        plt.hlines([0.3, 0.4, 0.5], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
        plt.ylim((0.25, 0.6))
      plt.legend(loc='lower right')

      plt.savefig(f'apr4_{"var_" if plot_var else ""}plot_subsampling_{d}_{e}.png')


def apr6_replot_nonprivate(plot_var=False):
  queried_setups = ['real_data', 'dpmerf-nonprivate', 'dpcgan-nonprivate', 'mar19_nonp_ae']
  setup_names = ['real data', 'DP-MERF non-DP', 'DP-CGAN non-DP', 'DP-MERF-AE non-DP']

  metric = 'accuracies'
  # metric = 'f1_scores'
  data_ids = ['d', 'f']
  sb_np_array = collect_results()['sb_np']

  # digit plot
  for d_id in data_ids:
    plt.figure(), plt.title(f'data: {"MNIST" if d_id == "d" else "FashionMNIST"}'), plt.xscale('log')
    # plt.xticks(DEFAULT_RATIOS[::-1], [str(k*100) for k in DEFAULT_RATIOS[::-1]])
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = sb_np_array.get({'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

      sub_mat = np.mean(sub_mat, axis=1)  # average over models

      if plot_var:
        plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.mean(sub_mat, axis=1)  # average over runs
        plt.plot(DEFAULT_RATIOS, sub_mat, label=s, color=DEFAULT_COLORS[s_idx])  # do show 1.0

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    if d_id == 'd':
      plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
      plt.ylim((0.4, 0.9))
    else:
      plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8])
      plt.ylim((0.4, 0.8))
    plt.legend(loc='upper left')
    plt.savefig(f'plots/apr4_nonp_{"var_" if plot_var else ""}{d_id}_{metric}.png')


def apr6_plot_overfit_conv(plot_var=False):
  queried_setups = ['apr4_sr_conv_sig_0', 'apr4_sr_conv_sig_2.5', 'apr4_sr_conv_sig_5',
                    'apr4_sr_conv_sig_10', 'apr4_sr_conv_sig_25', 'apr4_sr_conv_sig_50']
  setup_names = ['non-DP', 'eps=2', 'eps=1', 'eps=0.5', 'eps=0.2', 'eps=0.1']


  sub_ratios = [1.0, 0.1, 0.01, 0.001]
  data_used = ['60k', '6k', '600', '60']
  metric = 'accuracies'
  data_ids = ['d', 'f']
  sb_np_array = collect_results()['sr_conv_apr4']

  # digit plot
  for d_id in data_ids:
    plt.figure()
    plt.title(f'DP-MERF single release + convolutional generator: {"MNIST" if d_id == "d" else "FashionMNIST"}')
    plt.xscale('log')
    # plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])
    plt.xticks(sub_ratios[::-1], data_used[::-1])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = sb_np_array.get({'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      if plot_var:
        plot_with_variance(sub_ratios, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(sub_ratios, sub_mat, label=s, color=DEFAULT_COLORS[s_idx])  # do show 1.0

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.savefig(f'plots/apr4_sr_conv_{"var_" if plot_var else ""}{d_id}_{metric}.png')


def apr6_plot_better_conv(plot_var=False):
  queried_setups = ['real_data', 'dpcgan', 'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5', 'apr6_sr_conv_sig_25']
  setup_names = ['real data',
                 'DP-CGAN $\epsilon=9.6$', 'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                 'DP-MERF $\epsilon=0.2$']

  # metric = 'accuracies'
  metric = 'accuracies'
  data_ids = ['d', 'f']
  ar_dict = collect_results()
  sr_conv_array = ar_dict['sr_conv_apr6']
  sb_array = ar_dict['sb']
  merged_array = sr_conv_array.merge(sb_array, merge_dim='setups')

  # digit plot
  for d_id in data_ids:
    plt.figure()
    # plt.title(f'DP-MERF single release + convolutional generator: {"MNIST" if d_id == "d" else "FashionMNIST"}')
    plt.xscale('log')
    # plt.xticks(DEFAULT_RATIOS[::-1], [str(k*100) for k in DEFAULT_RATIOS[::-1]])
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    print('data', d_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

      print(f'setup: {s}')
      print(sub_mat.shape)
      print(np.mean(sub_mat, axis=2)[0])  # avg over runs

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      print(np.mean(sub_mat, axis=1)[0])  # avg over runs
      if plot_var:
        plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(DEFAULT_RATIOS, sub_mat, label=s, color=DEFAULT_COLORS[s_idx])  # do show 1.0

        print(f'mean values for setting {s}, data {d_id}:', sub_mat)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    if d_id == 'd':
      pass
      plt.yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
      plt.hlines([0.45, 0.5, 0.55, 0.6, 0.65], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.ylim((0.40, 0.7))
    else:
      pass
      plt.yticks([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
      plt.hlines([0.4, 0.45, 0.5, 0.55, 0.6], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.ylim((0.35, 0.65))
    # plt.legend(loc='upper left')
    plt.legend(loc='lower center')
    plt.savefig(f'plots/jun2_sr_conv_{"var_" if plot_var else ""}{d_id}_{metric}_with_dpcgan_notitle.png')


def jan7_plot_better_conv_plus_full_mmd(plot_var=False):
  queried_setups = ['real_data', 'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5', 'apr6_sr_conv_sig_25',
                    'full_mmd']
  setup_names = ['real data', 'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$', 'DP-MERF $\epsilon=0.2$',
                 'full MMD $\epsilon=\infty$']
  metric = 'accuracies'
  data_ids = ['d', 'f']
  ar_dict = collect_results()
  sr_conv_array = ar_dict['sr_conv_apr6']
  sb_array = ar_dict['sb']
  full_mmd_array = ar_dict['full_mmd_jan7']
  print(sr_conv_array.array.shape, sb_array.array.shape, full_mmd_array.array.shape)
  merged_array = sr_conv_array.merge(sb_array, merge_dim='setups')
  print(merged_array.array.shape)
  merged_array = merged_array.merge(full_mmd_array, merge_dim='setups')

  # digit plot
  for d_id in data_ids:
    plt.figure()
    # plt.title(f'DP-MERF single release + convolutional generator: {"MNIST" if d_id == "d" else "FashionMNIST"}')
    plt.xscale('log')
    # plt.xticks(DEFAULT_RATIOS[::-1], [str(k*100) for k in DEFAULT_RATIOS[::-1]])
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    print('data', d_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

      print(f'setup: {s}')
      print(sub_mat.shape)
      print(np.mean(sub_mat, axis=2)[0])  # avg over runs

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      print(np.mean(sub_mat, axis=1)[0])  # avg over runs
      if plot_var:
        plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(DEFAULT_RATIOS, sub_mat, label=s, color=DEFAULT_COLORS[s_idx])  # do show 1.0

        print(f'mean values for setting {s}, data {d_id}:', sub_mat)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    if d_id == 'd':
      pass
      plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
      plt.hlines([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.ylim((0.45, 0.9))
    else:
      pass
      plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
      plt.hlines([0.5, 0.55, 0.6, 0.65, 0.7, 0.75], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.ylim((0.45, 0.8))
    # plt.legend(loc='upper left')
    plt.legend(loc='upper left')
    plt.savefig(f'plots/jan7_sr_conv_{"var_" if plot_var else ""}{d_id}_{metric}_with_full_mmd.png')


def apr23_fashion_merf_plus_full_mmd_and_mehp(plot_var=False):
  queried_setups = ['real_data',
                    'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5',
                    # 'apr6_sr_conv_sig_25',
                    'full_mmd',
                    'mehp_nonDP', 'mehp_eps=1']
  setup_names = ['real data',
                 # 'DP-CGAN $\epsilon=9.6$',
                 'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                 # 'DP-MERF $\epsilon=0.2$',
                 'full MMD $\epsilon=\infty$',
                 'DP-MEHP $\epsilon=\infty$', 'DP-MEHP $\epsilon=1$']

  metric = 'accuracies'
  data_ids = ['f']

  ar_dict = collect_results()
  sr_conv_array = ar_dict['sr_conv_apr6']
  adaboost_array = ar_dict['may10_merf_adaboost']
  # print(sr_conv_array.array[:, 1:, :, 7, :, 0].shape, sr_conv_array.array[:, :, :, 0, :, 0].shape)
  sr_conv_array.array[:, 1:, :, 7, :, 0] = adaboost_array.array[:, :, :, 0, :, 0]

  # assert 1%1==1
  sb_array = ar_dict['sb']
  real_adaboost_array = ar_dict['may10_real_adaboost']
  print(sb_array.array.shape, real_adaboost_array.array.shape)
  print(sb_array.dim_names)
  for d in sb_array.dim_names:
    print(d, sb_array.idx_names[d])
  for d in real_adaboost_array.dim_names:
    print(d, real_adaboost_array.idx_names[d])
  sb_array.array[:, 0, :, 7, :, 0] = real_adaboost_array.array[:, 0, :, 0, :, 0]

  full_mmd_array = ar_dict['full_mmd_jan7']
  mehp_array = ar_dict['mehp_fmnist_apr23']

  merged_array = sr_conv_array.merge(sb_array, merge_dim='setups')
  merged_array = merged_array.merge(full_mmd_array, merge_dim='setups')
  merged_array = merged_array.merge(mehp_array, merge_dim='setups')

  # digit plot
  for d_id in data_ids:
    plt.figure()
    plt.xscale('log')
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    print('data', d_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

      print(f'setup: {s}')
      print(sub_mat.shape)
      print(np.mean(sub_mat, axis=2)[0])  # avg over runs

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      print(np.mean(sub_mat, axis=1)[0])  # avg over runs
      if plot_var:
        plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx], color=DEFAULT_COLORS[s_idx])  # do show 1.0

        print(f'mean values for setting {s}, data {d_id}:', sub_mat)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    if d_id == 'd':
      pass
      plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
      plt.hlines([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.ylim((0.45, 0.9))
    else:
      pass
      plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
      plt.hlines([0.5, 0.55, 0.6, 0.65, 0.7, 0.75], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
      plt.ylim((0.45, 0.8))
    # plt.legend(loc='upper left')
    plt.legend(loc='lower right')
    plt.savefig(f'plots/may10_{"var_" if plot_var else ""}fashion_merf_plus_full_mmd_and_mehp.png')

    for m_idx, model in enumerate(DEFAULT_MODELS):
      plt.figure()
      plt.xscale('log')
      plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
      for s_idx, s in enumerate(queried_setups):
        sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': [model], 'eval_metrics': [metric]})
        print(sub_mat.shape)
        print(np.mean(sub_mat, axis=1)[0])  # avg over runs
        if plot_var:
          plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
        else:
          sub_mat = np.median(sub_mat, axis=1)
          plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx], color=DEFAULT_COLORS[s_idx])  # do show 1.0
          print(f'median values for setting {s}, data {d_id}:', sub_mat)

      plt.xlabel('# samples generated')
      plt.ylabel('accuracy')
      plt.legend(loc='lower right')
      plt.savefig(f'plots/may10_{"var_" if plot_var else ""}fashion_plot_{model}.png')

    plt.figure()
    plt.xscale('log')
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s],
                                  'models': ['adaboost'], 'eval_metrics': ['accuracies']})
      print(sub_mat.shape)
      print(np.mean(sub_mat, axis=1)[0])  # avg over runs
      if plot_var:
        plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx], color=DEFAULT_COLORS[s_idx])  # do show 1.0
        print(f'median values for setting {s}, data {d_id}:', sub_mat)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.savefig(f'plots/may10_{"var_" if plot_var else ""}fashion_plot_{model}.png')


def may20_digits_merf_plus_full_mmd_and_mehp(plot_var=False):
  metric = 'accuracies'
  data_ids = ['d']
  ar_dict = collect_results()
  sr_conv_array = ar_dict['sr_conv_apr6']
  merf_adaboost_array = ar_dict['may10_merf_adaboost']
  sr_conv_array.array[:, 1:, :, 7, :, 0] = merf_adaboost_array.array[:, :, :, 0, :, 0]

  sb_array = ar_dict['sb']
  real_adaboost_array = ar_dict['may10_real_adaboost']
  sb_array.array[:, 0, :, 7, :, 0] = real_adaboost_array.array[:, 0, :, 0, :, 0]

  full_mmd_array = ar_dict['full_mmd_jan7']
  mehp_array = ar_dict['mehp_dmnist_may20_fc']
  merged_array = sr_conv_array.merge(sb_array, merge_dim='setups')
  merged_array = merged_array.merge(full_mmd_array, merge_dim='setups')
  merged_array = merged_array.merge(mehp_array, merge_dim='setups')

  orders = [20, 50, 100, 200, 500, 1000]
  # digit plot
  for d_id in data_ids:
    for o_idx, order in enumerate(orders):
      queried_setups = ['real_data',
                        'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5',
                        # 'apr6_sr_conv_sig_25',
                        'full_mmd',
                        f'fc mehp non-DP order{order}', f'fc mehp eps=1 order{order}']
      setup_names = ['real data',
                     'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                     # 'DP-MERF $\epsilon=0.2$',
                     'full MMD $\epsilon=\infty$',
                     f'mehp $\epsilon=\infty$ order{order}', f'mehp $\epsilon=1$ order{order}']
      plt.figure()
      plt.xscale('log')
      plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
      print('data', d_id)
      for s_idx, s in enumerate(queried_setups):
        sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

        print(f'setup: {s}')
        print(sub_mat.shape)
        print(np.mean(sub_mat, axis=2)[0])  # avg over runs

        sub_mat = np.mean(sub_mat, axis=1)  # average over models
        print(np.mean(sub_mat, axis=1)[0])  # avg over runs
        if plot_var:
          plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
        else:
          sub_mat = np.median(sub_mat, axis=1)
          plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx], color=DEFAULT_COLORS[s_idx])  # do show 1.0

          print(f'mean values for setting {s}, data {d_id}:', sub_mat)

      plt.xlabel('# samples generated')
      plt.ylabel('accuracy')

      plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
      plt.hlines([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0],
                 linestyles='dotted')
      plt.ylim((0.45, 0.9))
      plt.legend(loc='lower right')
      plt.savefig(f'plots/may10_order{order}_{"var_" if plot_var else ""}digit_merf_plus_full_mmd_and_mehp.png')

      for m_idx, model in enumerate(DEFAULT_MODELS):
        plt.figure()
        plt.xscale('log')
        plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
        for s_idx, s in enumerate(queried_setups):
          sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': [model], 'eval_metrics': [metric]})
          print(sub_mat.shape)
          print(np.mean(sub_mat, axis=1)[0])  # avg over runs
          if plot_var:
            plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
          else:
            sub_mat = np.median(sub_mat, axis=1)
            plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx], color=DEFAULT_COLORS[s_idx])  # do show 1.0
            print(f'median values for setting {s}, data {d_id}:', sub_mat)

        plt.xlabel('# samples generated')
        plt.ylabel('accuracy')
        plt.legend(loc='lower right')
        plt.savefig(f'plots/may10_order{order}_{"var_" if plot_var else ""}digit_plot_{model}.png')


# def may20_dmnist_mehp_order_comp(plot_var=False, private=False):
#
#   metric = 'accuracies'
#   data_ids = ['d']
#   ar_dict = collect_results()
#   sr_conv_array = ar_dict['sr_conv_apr6']
#   sb_array = ar_dict['sb']
#   full_mmd_array = ar_dict['full_mmd_jan7']
#   mehp_array = ar_dict['mehp_dmnist_may20_fc']
#   merged_array = sr_conv_array.merge([sb_array, full_mmd_array, mehp_array], merge_dim='setups')
#
#   var_str = "var_" if plot_var else ""
#   dp_str = 'eps=1_' if private else 'nonDP_'
#
#   orders = [20, 50, 100, 200, 500, 1000]
#   if private:
#     queried_setups = [f'fc mehp eps=1 order{o}' for o in orders]
#     setup_names = [f'mehp $\epsilon=1$ order{o}' for o in orders]
#   else:
#     queried_setups = [f'fc mehp non-DP order{o}' for o in orders]
#     setup_names = [f'mehp $\epsilon=\infty$ order{o}' for o in orders]
#   # digit plot
#   for d_id in data_ids:
#
#       plt.figure()
#       plt.xscale('log')
#       plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
#       print('data', d_id)
#       for s_idx, s in enumerate(queried_setups):
#         sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s],
#                                     'models': DEFAULT_MODELS, 'eval_metrics': [metric]})
#
#         print(f'setup: {s}')
#         print(sub_mat.shape)
#         print(np.mean(sub_mat, axis=2)[0])  # avg over runs
#
#         sub_mat = np.mean(sub_mat, axis=1)  # average over models
#         print(np.mean(sub_mat, axis=1)[0])  # avg over runs
#         if plot_var:
#           plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx],
#                              label=setup_names[s_idx])
#         else:
#           sub_mat = np.median(sub_mat, axis=1)
#           plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx],
#                    color=DEFAULT_COLORS[s_idx])  # do show 1.0
#
#           print(f'mean values for setting {s}, data {d_id}:', sub_mat)
#
#       plt.xlabel('# samples generated')
#       plt.ylabel('accuracy')
#
#       plt.yticks([0.55, 0.6, 0.65, 0.7, 0.75])
#       plt.hlines([0.6, 0.65, 0.7],
#                  xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
#       plt.ylim((0.55, 0.75))
#       plt.legend(loc='lower right')
#       plt.savefig(f'plots/may10_order_comp_{dp_str}{var_str}digit_plot.png')
#       plt.close()
#
#       for m_idx, model in enumerate(DEFAULT_MODELS):
#         plt.figure()
#         plt.xscale('log')
#         plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
#         for s_idx, s in enumerate(queried_setups):
#           sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': [model],
#                                       'eval_metrics': [metric]})
#
#           if plot_var:
#             plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx],
#                                label=setup_names[s_idx])
#           else:
#             sub_mat = np.median(sub_mat, axis=1)
#             plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx],
#                      color=DEFAULT_COLORS[s_idx])  # do show 1.0
#             print(f'median values for setting {s}, data {d_id}:', sub_mat)
#
#         plt.xlabel('# samples generated')
#         plt.ylabel('accuracy')
#         plt.legend(loc='lower right')
#         plt.savefig(f'plots/may10_order_comp_{dp_str}{var_str}digit_plot_{model}.png')
#         plt.close()


def may20_mehp_subsampled():
  metric = 'accuracies'
  data_ids = ['d', 'f']

  for data_id in data_ids:
    if data_id == 'd':
      save_str = 'digit'
      data_str = 'mehp_dmnist_may20_fc'
      setups = ['fc mehp non-DP order100', 'fc mehp eps=1 order100']
      y_lims = (0.4, 0.9)
      y_step = 0.05

    else:
      save_str = 'fashion'
      data_str = 'mehp_fmnist_apr23'
      setups = ['mehp_nonDP', 'mehp_eps=1']
      y_lims = (0.35, 0.8)
      y_step = 0.05

    plot_data_dict = dict()
    ar_dict = collect_results()
    sr_conv_array = ar_dict['sr_conv_apr6']
    merf_adaboost_array = ar_dict['may10_merf_adaboost']
    sr_conv_array.array[:, 1:, :, 7, :, 0] = merf_adaboost_array.array[:, :, :, 0, :, 0]

    sb_array = ar_dict['sb']
    real_adaboost_array = ar_dict['may10_real_adaboost']
    sb_array.array[:, 0, :, 7, :, 0] = real_adaboost_array.array[:, 0, :, 0, :, 0]

    merged_array = sr_conv_array.merge([sb_array, ar_dict['full_mmd_jan7'],
                                        ar_dict[data_str],
                                        ar_dict['may14_gswgan'],
                                        ar_dict['may27_dpgan']],
                                       merge_dim='setups')

    queried_setups = ['real_data', 'full_mmd',
                      setups[0], setups[1],
                      'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5',
                      'gswgan', 'dpcgan', 'may27_dpgan']
    setup_names = ['real data', 'full MMD $\epsilon=\infty$',
                   'DP-HP (ours) $\epsilon=\infty$', 'DP-HP (ours) $\epsilon=1$',
                   'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                   'GS-WGAN $\epsilon=10$', 'DP-CGAN $\epsilon=9.6$', 'DP-GAN $\epsilon=9.6$']
    plt.figure()
    ax = plt.subplot(111)
    plt.xscale('log')
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    print('data', data_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [data_id], 'setups': [s], 'models': DEFAULT_MODELS,
                                  'eval_metrics': [metric]})

      print(f'setup: {s}')

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      mean_y_100 = np.mean(sub_mat, axis=1)[0]
      mean_y_opt = np.max(np.mean(sub_mat, axis=1))
      sdev_y_100 = np.std(sub_mat, axis=1)[0]
      print(f'acc mean: {mean_y_100}, sdev: {sdev_y_100}, opt: {mean_y_opt}')

      means_y, sdevs_y = plot_with_variance(DEFAULT_RATIOS, sub_mat,
                                            color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])
      plot_data_dict[str(s_idx)] = (setup_names[s_idx], means_y, sdevs_y)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')

    y_ticks = np.arange(y_lims[0], y_lims[1] + y_step, y_step)
    h_lines = np.arange(y_lims[0] + y_step, y_lims[1], y_step)
    plt.yticks(y_ticks)
    plt.hlines(h_lines, xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
    plt.ylim(y_lims)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'{save_str.capitalize()} MNIST downstream accuracy under subsampling', loc='center', pad=10.)
    plt.savefig(f'plots/may27_order100_var_{save_str}_mehp.png')
    np.savez(f'plots/may27_order100_var_{save_str}_mehp_values.npz', **plot_data_dict)


def may20_dmnist_mehp_order_comp(plot_var=False, private=False):
  metric = 'accuracies'
  data_ids = ['d']
  ad = collect_results()
  merged_array = ad['sr_conv_apr6'].merge([ad['sb'], ad['full_mmd_jan7'], ad['mehp_dmnist_may20_fc']], merge_dim='setups')

  var_str = "var_" if plot_var else ""
  dp_str = 'eps=1_' if private else 'nonDP_'

  orders = [20, 50, 100, 200, 500, 1000]
  if private:
    queried_setups = [f'fc mehp eps=1 order{o}' for o in orders]
    setup_names = [f'order = {o}' for o in orders]
  else:
    queried_setups = [f'fc mehp non-DP order{o}' for o in orders]
    setup_names = [f'order = {o}' for o in orders]
  # digit plot
  for d_id in data_ids:

    plt.figure()
    plt.xscale('log')
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    print('data', d_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get(
        {'data_ids': [d_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

      print(f'setup: {s}')
      print(sub_mat.shape)
      print(np.mean(sub_mat, axis=2)[0])  # avg over runs

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      print(np.mean(sub_mat, axis=1)[0])  # avg over runs
      if plot_var:
        plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx],
                           label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(DEFAULT_RATIOS, sub_mat, label=setup_names[s_idx],
                 color=DEFAULT_COLORS[s_idx])  # do show 1.0

        print(f'mean values for setting {s}, data {d_id}:', sub_mat)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')

    y_lims = (0.5, 0.75)
    y_step = 0.05
    y_ticks = np.arange(y_lims[0], y_lims[1] + y_step, y_step)
    h_lines = np.arange(y_lims[0] + y_step, y_lims[1], y_step)
    plt.yticks(y_ticks)
    plt.hlines(h_lines, xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
    plt.ylim(y_lims)
    eps_str = "1" if private else "$\infty$"
    plt.legend(loc='lower right')
    plt.title(f'downstream accuracy by hermite polynomial order ($\epsilon=$' + eps_str + ')',
              loc='center', pad=10.)
    plt.savefig(f'plots/may17_order_comp_{dp_str}{var_str}digit_plot.png')
    plt.close()


def may17_mehp_by_model():
  data_ids = ['d', 'f']

  for data_id in data_ids:
    if data_id == 'd':
      save_str = 'digit'
      data_str = 'mehp_dmnist_may20_fc'
      setups = ['fc mehp non-DP order100', 'fc mehp eps=1 order100']
      y_lims = (0.2, 1.0)
      y_step = 0.1

    else:
      save_str = 'fashion'
      data_str = 'mehp_fmnist_apr23'
      setups = ['mehp_nonDP', 'mehp_eps=1']
      y_lims = (0.2, 0.9)
      y_step = 0.1

    plot_data_dict = dict()  # saves exact data used for plotting
    ar_dict = collect_results()
    sr_conv_array = ar_dict['sr_conv_apr6']
    merf_adaboost_array = ar_dict['may10_merf_adaboost']
    sr_conv_array.array[:, 1:, :, 7, :, 0] = merf_adaboost_array.array[:, :, :, 0, :, 0]

    sb_array = ar_dict['sb']
    real_adaboost_array = ar_dict['may10_real_adaboost']
    sb_array.array[:, 0, :, 7, :, 0] = real_adaboost_array.array[:, 0, :, 0, :, 0]

    merged_array = sr_conv_array.merge([sb_array, ar_dict['full_mmd_jan7'],
                                        ar_dict[data_str],
                                        ar_dict['may14_gswgan']],
                                       merge_dim='setups')

    queried_setups = ['real_data', 'full_mmd',
                      setups[0], setups[1],
                      'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5',
                      'gswgan', 'dpcgan']
    setup_names = ['real data', 'full MMD $\epsilon=\infty$',
                   'DP-HP (ours) $\epsilon=\infty$', 'DP-HP (ours) $\epsilon=1$',
                   'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                   'GS-WGAN $\epsilon=10$', 'DP-CGAN $\epsilon=9.6$']
    plt.figure()
    ax = plt.subplot(111)
    plt.xticks(list(range(12)) + [13], DEFAULT_MODELS + ['MEAN'])
    plt.xticks(rotation=45, ha='right')
    print('data', data_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [data_id], 'setups': [s],
                                  'models': DEFAULT_MODELS, 'eval_metrics': ['accuracies']})
      print(f'setup: {s}')
      print(sub_mat.shape)

      sub_mat = sub_mat[0, :, :]  # select full data exp
      print(sub_mat.shape)

      means_y = np.mean(sub_mat, axis=1)
      means_y_plus_mean = np.concatenate([means_y, np.mean(means_y, keepdims=True)])
      plt.plot(list(range(12)) + [13], means_y_plus_mean, 'o', label=setup_names[s_idx],
               color=DEFAULT_COLORS[s_idx])
      plot_data_dict[str(s_idx)] = (setup_names[s_idx], means_y_plus_mean)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')

    y_ticks = np.arange(y_lims[0], y_lims[1] + y_step, y_step)
    h_lines = np.arange(y_lims[0] + y_step, y_lims[1], y_step)
    plt.yticks(y_ticks)
    plt.hlines(h_lines, xmin=0, xmax=13, linestyles='dotted')
    plt.vlines(12, ymin=0.2, ymax=1.)
    plt.ylim(y_lims)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.75, box.height * 0.95])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'{save_str.capitalize()} MNIST downstream accuracy by model', loc='center', pad=10.)
    plt.savefig(f'plots/may17_{save_str}_mehp_by_model.png')
    np.savez(f'plots/may17_{save_str}_mehp_by_model_values.npz', **plot_data_dict)


def sep17_mehp_subsampled_from_dict():
  """
  plots accuracy with variance over subsampling ratios averaged across models
  plot_data_dict is a dict with keys '0', '1', '2'... (to ensure plotting in the right order)
  each value of plot_data_dict is a tuple (name, means, sdevs), where
  - name is the name of the setup (eg. 'DP-HP eps=1') to be used as label
  - means is a vector of mean accuracies across all downstream models and 5 seeds each
  - sdevs is a vector of accuracy standard deviations across the same models and seeds
  the subsampling ratios, based on a full dataset size of 60.000 are
  DEFAULT_RATIOS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

  plotting can be extended by adding tuples to the plot_data_dict and possibly changing the keys
  currently the function assumes that keys are all strings of ascending integers.
  """
  data_ids = ['d', 'f']

  for data_id in data_ids:
    if data_id == 'd':
      save_str = 'digit'
      y_lims = (0.4, 0.9)
      y_step = 0.05

    else:
      save_str = 'fashion'
      y_lims = (0.35, 0.8)
      y_step = 0.05

    plot_data_dict = np.load(f'plots/may27_order100_var_{save_str}_mehp_values.npz',
                             allow_pickle=True)

    plt.figure()
    ax = plt.subplot(111)
    plt.xscale('log')
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    for idx in range(len(plot_data_dict.keys())):  # ensures right order, as keys are unsorted
      s_name, means_y, sdevs_y = plot_data_dict[str(idx)]
      plot_with_variance_given_mean_and_sdev(DEFAULT_RATIOS, means_y, sdevs_y,
                                             color=DEFAULT_COLORS[idx], label=s_name, alpha=0.1)

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')

    y_ticks = np.arange(y_lims[0], y_lims[1] + y_step, y_step)
    h_lines = np.arange(y_lims[0] + y_step, y_lims[1], y_step)
    plt.yticks(y_ticks)
    plt.hlines(h_lines, xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
    plt.ylim(y_lims)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'{save_str.capitalize()} MNIST downstream accuracy under subsampling', loc='center', pad=10.)
    plt.savefig(f'plots/may27_order100_var_{save_str}_mehp_redo.png')


def sep17_mehp_by_model_from_dict():
  """
  plots accuracy averaged across five seeds separately for each model and on average
  plot_data_dict is a dict with keys '0', '1', '2'... (to ensure plotting in the right order)
  each value of plot_data_dict is a tuple (name, means), where
  - name is the name of the setup (eg. 'DP-HP eps=1') to be used as label
  - means is a vector of mean accuracies across 5 seeds for each model, followed by the overall mean
  The order of downstream models is given by:
  DEFAULT_MODELS = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc',
                    'decision_tree', 'lda', 'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  plotting can be extended by adding tuples to the plot_data_dict and possibly changing the keys
  currently the function assumes that keys are all strings of ascending integers.
  """
  data_ids = ['d', 'f']

  for data_id in data_ids:
    if data_id == 'd':
      save_str = 'digit'
      y_lims = (0.2, 1.0)
      y_step = 0.1

    else:
      save_str = 'fashion'
      y_lims = (0.2, 0.9)
      y_step = 0.1

    # np.savez(f'plots/may17_{save_str}_mehp_by_model_values.npz', **plot_data_dict)
    plot_data_dict = np.load(f'plots/may17_{save_str}_mehp_by_model_values.npz',
                             allow_pickle=True)

    plt.figure()
    ax = plt.subplot(111)
    plt.xticks(list(range(12)) + [13], DEFAULT_MODELS + ['MEAN'])
    plt.xticks(rotation=45, ha='right')
    print('data', data_id)
    for idx in range(len(plot_data_dict.keys())):  # ensures right order, as keys are unsorted
      s_name, means_y = plot_data_dict[str(idx)]
      plt.plot(list(range(12)) + [13], means_y, 'o', label=s_name, color=DEFAULT_COLORS[idx])

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')

    y_ticks = np.arange(y_lims[0], y_lims[1] + y_step, y_step)
    h_lines = np.arange(y_lims[0] + y_step, y_lims[1], y_step)
    plt.yticks(y_ticks)
    plt.hlines(h_lines, xmin=0, xmax=13, linestyles='dotted')
    plt.vlines(12, ymin=0.2, ymax=1.)
    plt.ylim(y_lims)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.75, box.height * 0.95])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'{save_str.capitalize()} MNIST downstream accuracy by model', loc='center', pad=10.)
    plt.savefig(f'plots/may17_{save_str}_mehp_by_model_redo.png')


if __name__ == '__main__':
  # apr23_fashion_merf_plus_full_mmd_and_mehp(plot_var=True)
  # apr27_digits_merf_plus_full_mmd_and_mehp(plot_var=True)
  # apr27_dmnist_mehp_order_comp(plot_var=True, private=True)
  # apr27_dmnist_mehp_order_comp(plot_var=True, private=False)
  # may17_mehp_subsampled(plot_var=True)
  # may17_dmnist_mehp_order_comp(plot_var=True, private=True)
  # may17_dmnist_mehp_order_comp(plot_var=True, private=False)
  # may17_mehp_subsampled(plot_var=True)
  # may20_digits_merf_plus_full_mmd_and_mehp(plot_var=True)
  # may20_dmnist_mehp_order_comp(plot_var=True, private=True)
  # may20_dmnist_mehp_order_comp(plot_var=True)

  # may17_mehp_by_model()
  # may20_mehp_subsampled()

  sep17_mehp_subsampled_from_dict()
  sep17_mehp_by_model_from_dict()
