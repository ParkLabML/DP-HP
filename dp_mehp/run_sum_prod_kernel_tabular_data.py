### this script is to test DP-HP on tabular data


import numpy as np
import os
import time
from autodp import privacy_calibrator
import matplotlib
#matplotlib.use('Agg')
import argparse


from all_aux_files_tab_data import data_loading
from all_aux_files_tab_data import Generative_Model_homogeneous_data
from all_aux_files_tab_data import Generative_Model_heterogeneous_data
from all_aux_files_tab_data import test_models
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import torch

from all_aux_files_tab_data import find_rho_tab, ME_with_HP_tab, ME_with_HP_prod, heuristic_for_length_scale
#from all_aux_files import log_args


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='intrusion', help='choose among cervical, adult, census, intrusion, covtype, epileptic, credit, isolet')

    # OPTIMIZATION
    parser.add_argument("--batch-rate", type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

    # DP SPEC
    parser.add_argument('--is-private', type=int, default=1, help='produces a DP mean embedding of data')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta in (epsilon, delta)-DP')

    # OTHERS
    parser.add_argument("--undersampled-rate", type=float, default=0.3)
    parser.add_argument("--normalize-data", action='store_true', default=False)

    # Downstream models
    parser.add_argument('--classifiers', nargs='+', type=int, help='list of integers',
                      default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # Kernel related arguments
    parser.add_argument('--combined-kernel', default=True, help='If true use both sum and product kernels. If false, use sum kernel only')
    parser.add_argument('--order-hermite', type=int, default=100, help='HP order for sum kernel')
    parser.add_argument('--prod-dimension', type=int, default=5, help='select the number of dimensions for product kernel')
    parser.add_argument('--order-hermite-prod', type=int, default=10, help='order of HP for approximating the product kernel')
    parser.add_argument('--gamma', type=float, default=0.01, help='a weight to apply to the ME under the product kernel')

    # assign different epsilons to ME_prod and ME_sum
    parser.add_argument('--split-eps', default=True,
                      help='we split epsilon into two parts according to split-eps-rate')
    parser.add_argument('--split-eps-rate', type=float, default=0.6,
                      help='a proportion of epsilon assigned to ME_sum and the rest is assigned to ME_prod')


    
    ar = parser.parse_args()

    return ar


def preprocess_args(ar):

    """ name the directories """
    base_dir = 'logs/gen/'

    if ar.combined_kernel:

        log_name = ar.data_name + '_' + 'seed=' + str(ar.seed) +  \
               '_' + 'sum_order_hermite=' + str(ar.order_hermite) + '_' + 'prod_subsample_dim' + str(ar.prod_dimension) + '_' + 'prod_order' + str(ar.order_hermite_prod) + '_' + 'private=' + str(ar.is_private) + '_' \
               + 'epsilon=' + str(ar.epsilon) + '_' + 'delta=' + str(ar.delta) + '_' + 'br=' + str(ar.batch_rate) + '_' + 'lr=' + str(ar.lr) + '_' \
               + 'n_epoch=' + str(ar.epochs) + '_' + 'undersam_rate=' + str(ar.undersampled_rate)  + '_' + 'gamma_val' + str(ar.gamma) \
               + '_' + 'normalize_data' + str(ar.normalize_data) +'_' + 'combined_kernel='+str(ar.combined_kernel)

    else: # sum kernel alone

        log_name = ar.data_name + '_' + 'seed=' + str(ar.seed) +  \
               '_' + 'order=' + str(ar.order_hermite) + '_' + 'private=' + str(ar.is_private) + '_' \
               + 'epsilon=' + str(ar.epsilon) + '_' + 'delta=' + str(ar.delta) + '_' \
                + 'br=' + str(ar.batch_rate) + '_' + 'lr=' + str(ar.lr) + '_' \
               + 'n_epoch=' + str(ar.epochs) + '_' + 'undersam_rate=' + str(ar.undersampled_rate) \
               + '_' + 'normalize_data' + str(ar.normalize_data) +'_' + 'kernel_type'+str(ar.kernel_type)



    ar.log_name = log_name
    ar.log_dir = base_dir + log_name + '/'
    if not os.path.exists(ar.log_dir):
        os.makedirs(ar.log_dir)


def main(data_name, seed_num, order_hermite, batch_rate, n_epochs, undersampled_rate, order_hermite_prod, prod_dimension, gamma):

    # load settings
    ar = get_args()
    ar.data_name = data_name
    ar.seed = seed_num
    ar.order_hermite = order_hermite # HP order for approximating the sum kernel
    ar.batch_rate = batch_rate # this determines how many steps the algorithm runs in each epoch
    ar.epochs = n_epochs # for how many epochs the algorithm runs
    ar.undersampled_rate = undersampled_rate # for dealing with imbalance between classes in some datasets
    ar.order_hermite_prod = order_hermite_prod # HP order for approximating the product kernel
    ar.prod_dimension = prod_dimension # number of input dimensions subsampled in each epoch
    ar.gamma = gamma # weighs loss_prod
    
    preprocess_args(ar)
    #log_args(ar.log_dir, ar)

    torch.manual_seed(seed_num)

    if ar.is_private:
        epsilon = ar.epsilon
        delta = ar.delta

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)

    # specify heterogeneous dataset or not
    heterogeneous_datasets = ['cervical', 'adult', 'census', 'intrusion', 'covtype']
    homogeneous_datasets = ['epileptic','credit','isolet']

    """ Load data to test """
    X_train, X_test, y_train, y_test, n_classes, num_categorical_inputs, num_numerical_inputs = data_loading(data_name, ar.undersampled_rate, ar.seed)

    # one-hot encoding of labels.
    n, input_dim = X_train.shape

    # one hot encode the labels
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = np.expand_dims(y_train, 1)
    true_labels = onehot_encoder.fit_transform(y_train)

    # standardize the inputs
    if ar.normalize_data:
        print('normalizing the data')
        X_train = preprocessing.minmax_scale(X_train, feature_range=(0, 1), axis=0, copy=True)
        X_test = preprocessing.minmax_scale(X_test, feature_range=(0, 1), axis=0, copy=True)
    else:
        print('testing non-standardized data')


    ######################################
    # MODEL
    batch_size = np.int(np.round(batch_rate * n))
    input_size = 10 + 1
    hidden_size_1 = 4 * input_dim
    hidden_size_2 = 2 * input_dim
    output_size = input_dim

    if data_name in homogeneous_datasets:

        model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                    hidden_size_2=hidden_size_2,
                                                  output_size=output_size, dataset=data_name).to(device)

    else: # data_name in heterogeneous_datasets:

        model = Generative_Model_heterogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                    hidden_size_2=hidden_size_2,
                                                    output_size=output_size,
                                                    num_categorical_inputs=num_categorical_inputs,
                                                    num_numerical_inputs=num_numerical_inputs).to(device)

    """ set the scale length """
    sigma = heuristic_for_length_scale(X_train, input_dim)
    print('we use a separate length scale on each coordinate of the data using the median heuristic')
    sigma2 = sigma ** 2

    rho = find_rho_tab(sigma2)
    order = ar.order_hermite # HP order for the sum kernel

    ########## data mean embedding under the sum kernel ##########

    """ compute the weights """
    print('computing mean embedding of data (sum kernel): (1) compute the weights')
    unnormalized_weights = np.sum(true_labels, 0)
    weights = unnormalized_weights / np.sum(unnormalized_weights)  # weights = m_c / n
    print('\n weights without privatization are', weights, '\n')

    print('computing mean embedding of data (sum kernel): (2) compute the mean')
    data_embedding = torch.zeros(input_dim*(order+1), n_classes, device=device)
    for idx in range(n_classes):
        print(idx,'th-class')
        idx_data = X_train[y_train.squeeze()==idx,:]
        phi_data = ME_with_HP_tab(torch.Tensor(idx_data).to(device), order, rho, device, n)
        data_embedding[:,idx] = phi_data # this includes 1/n factor inside
        del phi_data
    print('done with computing mean embedding of data (sum kernel)')

    if ar.is_private:
        print("private")
        if ar.combined_kernel:
            if ar.split_eps:
                privacy_param_sum = privacy_calibrator.gaussian_mech(ar.split_eps_rate * ar.epsilon, delta * 0.5,
                                                                     k=2)
                print('Noise level sigma for sum kernel and weights =', privacy_param_sum['sigma'])
                privacy_param_prod = privacy_calibrator.gaussian_mech((1 - ar.split_eps_rate) * ar.epsilon,
                                                                      delta * 0.5, k=ar.epochs)
                print('Noise level sigma for prod kernel =', privacy_param_prod['sigma'])
            else:
                k = 2 + ar.epochs # because we add noise to weights (once), ME_sum (once) and ME_prod (in every epoch), where each ME takes two Gaussian mechanisms
                privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
                print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])
        else:
            k = 2  # because we add noise to the weights and means separately.
            privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
            print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])


        sensitivity_for_weights = np.sqrt(2) / n  # double check if this is sqrt(2) or 2
        if ar.split_eps:
            noise_std_for_weights = privacy_param_sum['sigma'] * sensitivity_for_weights
        else:
            noise_std_for_weights = privacy_param['sigma'] * sensitivity_for_weights
        weights = weights + np.random.randn(weights.shape[0]) * noise_std_for_weights
        weights[weights < 0] = 1e-3  # post-processing so that we don't have negative weights.
        weights = weights / sum(weights)  # post-processing so the sum of weights equals 1.
        print('weights after privatization are', weights)

        print('we add noise to the data mean embedding as the private flag is true')
        if ar.split_eps:
            std_sum = (2 * privacy_param_sum['sigma'] / n)
        else:
            std_sum = (2 * privacy_param['sigma'] / n)
        noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std_sum
        data_embedding = data_embedding + noise

    ### we want to ensure that data_embedding is divided by 1/m_c to deal with class imbalance in the following way
    data_embedding = data_embedding / torch.Tensor(weights).to(device)  # this means, 1/n * n/m_c, so 1/m_c


    """ Training """
    optimizer = torch.optim.Adam(list(model.parameters()), lr=ar.lr)
    num_iter = np.int(n / batch_size)
    # construct mini-batch dataset


    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()

        if ar.combined_kernel:
            order_prod = ar.order_hermite_prod
            dimensions_subsample = np.random.choice(input_dim, ar.prod_dimension, replace=False)
            # print("These are the dimensions sumsampled for the prod kernel: ", dimensions_subsample)
            rho_prod = rho[dimensions_subsample]  # same as the rho applied to the sum kernel, rho matched based on subsampled inputs
            prod_kernel_embedding_dim = pow(order_prod + 1, ar.prod_dimension)  # (C+1)**D_prod
            data_embedding_prod_kernel = torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)

            # print("Computing prod kernel mean embedding given a set of subsampled input dimensions at epoch {}".format(epoch))
            for idx_class in range(n_classes):
                chunk_size = 250
                for idx in range(n // chunk_size + 1):
                    idx_real_data = X_train[idx * chunk_size:(idx + 1) * chunk_size].astype(np.float32)
                    phi_data_prod_kernel = ME_with_HP_prod(torch.Tensor(idx_real_data[:, dimensions_subsample]).to(device),
                                                           order_prod, rho_prod,
                                                           device, n, prod_kernel_embedding_dim)
                    data_embedding_prod_kernel[:, idx_class] += phi_data_prod_kernel  # this has 1/n factor in it.

            if ar.is_private:
                # print('we add noise to the mean embedding prod kernel as is_private is set to True.')
                # Draw noise for the prod kernel mean emebdding as many times as epochs.
                if ar.split_eps:
                    std_prod = (2 * privacy_param_prod['sigma'] / n)
                else:
                    std_prod = (2 * privacy_param['sigma'] / n)
                noise_prod = torch.randn(data_embedding_prod_kernel.shape[0], data_embedding_prod_kernel.shape[1],
                                         device=device) * std_prod
                data_embedding_prod_kernel = data_embedding_prod_kernel + noise_prod

            ### we want to ensure that data_embedding is divided by 1/m_c to deal with class imbalance in the following way
            data_embedding_prod_kernel = data_embedding_prod_kernel / torch.Tensor(weights).to(device)  # this means, 1/n * n/m_c, so 1/m_c


        for i in range(num_iter):

            if data_name in homogeneous_datasets:  # In our case, if a dataset is homogeneous, then it is a binary dataset.

                label_input = torch.multinomial(torch.Tensor([weights]), batch_size, replacement=True).type(
                    torch.FloatTensor)
                label_input = label_input.transpose_(0, 1)
                label_input = label_input.squeeze()
                label_input = label_input.to(device)

                feature_input = torch.randn((batch_size, input_size - 1)).to(device)
                input_to_model = torch.cat((feature_input, label_input[:, None]), 1)

            else:  # heterogeneous data

                label_input = torch.multinomial(torch.Tensor([weights]), batch_size, replacement=True).type(
                        torch.FloatTensor)

                label_input = torch.cat((label_input, torch.arange(len(weights), out=torch.FloatTensor()).unsqueeze(0)),
                                        1)  # to avoid no labels
                label_input = label_input.transpose_(0, 1)
                label_input = label_input.squeeze()
                label_input = label_input.to(device)

                # (2) generate corresponding features
                feature_input = torch.randn((batch_size + len(weights), input_size - 1)).to(device)
                input_to_model = torch.cat((feature_input, label_input[:,None]), 1)

            """ (2) produce data """
            outputs = model(input_to_model)

            """ (3) compute synthetic data's mean embedding """
            weights_syn = torch.ones(n_classes) # weights = m_c / batch_size # to avoid dividing by 0
            syn_data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)
            if ar.combined_kernel:
                prod_kernel_embedding_dim = pow(order_prod + 1, ar.prod_dimension)  # (C+1)**D_prod
                synth_data_embedding_prod_kernel = torch.zeros((prod_kernel_embedding_dim, n_classes), device=device)

            for idx in range(n_classes):
                weights_syn[idx] = torch.sum(label_input == idx)
                idx_syn_data = outputs[label_input == idx]
                phi_syn_data = ME_with_HP_tab(idx_syn_data, order, rho, device, batch_size)
                syn_data_embedding[:, idx] = phi_syn_data  # this includes 1/n factor inside
                if ar.combined_kernel:
                    synth_data_embedding_prod_kernel[:, idx] = ME_with_HP_prod(idx_syn_data[:, dimensions_subsample],
                                                                               order_prod, rho_prod, device, batch_size,
                                                                               prod_kernel_embedding_dim)

            weights_syn = weights_syn / torch.sum(weights_syn)
            syn_data_embedding = syn_data_embedding / torch.Tensor(weights_syn).to(device)
            synth_data_embedding_prod_kernel = synth_data_embedding_prod_kernel / torch.Tensor(weights).to(device)  # this means, 1/n * n/m_c, so 1/m_c
            ########################################################################################

            ########################################################################################
            if ar.combined_kernel:
                loss_prod = torch.sum((data_embedding_prod_kernel - synth_data_embedding_prod_kernel) ** 2)
                loss_sum = torch.sum((data_embedding - syn_data_embedding) ** 2)
                loss = loss_sum + ar.gamma * loss_prod
            else:
                loss = torch.sum((data_embedding - syn_data_embedding)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
            if ar.combined_kernel:
                print('loss_sum :', loss_sum)
                print('loss_prod :', loss_prod)
                print('loss_prod * gamma :', loss_prod * ar.gamma)

                print(torch.norm(synth_data_embedding_prod_kernel))
                print(torch.norm(data_embedding_prod_kernel))

        # scheduler.step()


    """ Once the training step is over, we produce 60K samples and test on downstream tasks """
    """ now we save synthetic data of size 60K and test them on logistic regression """
    #######################################################################33
    if data_name in heterogeneous_datasets:

        """ draw final data samples """
        # (1) generate labels
        # if data_name == 'cervical':
        #     label_input = torch.multinomial(1 / n_classes * torch.ones(n_classes), n, replacement=True).type(
        #         torch.FloatTensor)
        #     label_input = label_input[None, :]
        # else:
        label_input = torch.multinomial(torch.Tensor([weights]), n, replacement=True).type(torch.FloatTensor)
        label_input = label_input.transpose_(0, 1)
        label_input = label_input.to(device)

        # (2) generate corresponding features
        feature_input = torch.randn((n, input_size - 1)).to(device)
        input_to_model = torch.cat((feature_input, label_input), 1)
        outputs = model(input_to_model)

        samp_input_features = outputs
        samp_labels = label_input

        # (3) round the categorial features
        output_numerical = outputs[:, 0:num_numerical_inputs]
        output_categorical = outputs[:, num_numerical_inputs:]
        output_categorical = torch.round(output_categorical)

        output_combined = torch.cat((output_numerical, output_categorical), 1)

        generated_input_features_final = output_combined.cpu().detach().numpy()
        generated_labels_final = label_input.cpu().detach().numpy()

        roc, prc = test_models(generated_input_features_final, generated_labels_final, X_test, y_test, n_classes, "generated", ar.classifiers, ar.data_name)


    else:  # homogeneous datasets

        """ draw final data samples """
        label_input = (1 * (torch.rand((n)) < weights[1])).type(torch.FloatTensor)
        # label_input = torch.multinomial(torch.Tensor([weights]), n, replacement=True).type(torch.FloatTensor)
        # label_input = torch.multinomial(1 / n_classes * torch.ones(n_classes), n, replacement=True).type(torch.FloatTensor)
        label_input = label_input.to(device)

        feature_input = torch.randn((n, input_size - 1)).to(device)
        input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
        outputs = model(input_to_model)

        samp_input_features = outputs

        label_input_t = torch.zeros((n, n_classes))
        idx_1 = (label_input == 1.).nonzero()[:, 0]
        idx_0 = (label_input == 0.).nonzero()[:, 0]
        label_input_t[idx_1, 1] = 1.
        label_input_t[idx_0, 0] = 1.

        samp_labels = label_input_t

        generated_input_features_final = samp_input_features.cpu().detach().numpy()
        generated_labels_final = samp_labels.cpu().detach().numpy()
        generated_labels = np.argmax(generated_labels_final, axis=1)

        roc, prc = test_models(generated_input_features_final, generated_labels, X_test, y_test, n_classes, "generated", ar.classifiers, ar.data_name)


    ####################################################
    """ saving results """
    dir_result = ar.log_dir + '/scores'
    np.save(dir_result + '_roc', roc)
    np.save(dir_result + '_prc', prc)
    np.save(dir_result + '_mean_roc', np.mean(roc))
    np.save(dir_result + '_mean_prc', np.mean(prc))

    """ saving synthetic data """
    dir_syn_data = ar.log_dir + '/synthetic_data'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)
    np.save(dir_syn_data + '/input_features', samp_input_features.detach().cpu().numpy())
    np.save(dir_syn_data + '/labels', samp_labels.detach().cpu().numpy())


    return roc, prc, ar.log_dir

if __name__ == '__main__':

    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
    ar = get_args()
    print(ar)

    for dataset in [ar.data_name]:

        print("\n\n")

        grid_insidefile=1
        if grid_insidefile:
            if dataset == 'adult':
                order_hermite_arg = [100] # HP order for sum kernel
                batch_rate = [0.1]
                how_many_epochs_arg = [20] #100
                undersampled_rate = [0.4] # to deal with class imbalance
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [5] # HP order for prod kernel
                prod_dimension_arg = [5] # subsampled input dimension for product kernel
                gamma_arg = [0.1]

            elif dataset == 'census':
                order_hermite_arg = [100] # HP order for sum kernel
                batch_rate = [0.4]
                how_many_epochs_arg = [50]
                undersampled_rate = [0.2] # to deal with class imbalance
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [5] # HP order for prod kernel
                prod_dimension_arg = [5] # subsampled input dimension for product kernel
                gamma_arg = [0.1]

            elif dataset == 'cervical':
                order_hermite_arg = [20]  # HP order for sum kernel
                batch_rate = [0.1]
                how_many_epochs_arg = [20]
                undersampled_rate = [0.3]  # to deal with class imbalance #1.0
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [13]  # HP order for prod kernel
                prod_dimension_arg = [5]  # subsampled input dimension for product kernel
                gamma_arg = [1.0]

            elif dataset == 'credit':
                order_hermite_arg = [20] # HP order for sum kernel
                batch_rate = [0.5]
                how_many_epochs_arg = [200]
                undersampled_rate = [0.002] # to deal with class imbalance
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [7] # HP order for prod kernel
                prod_dimension_arg = [5] # subsampled input dimension for product kernel
                gamma_arg = [1e-5]

            elif dataset == 'covtype':
                order_hermite_arg = [10] # HP order for sum kernel
                batch_rate = [0.01]
                how_many_epochs_arg = [200]
                undersampled_rate = [0.03] # to deal with class imbalance
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [7] # HP order for prod kernel
                prod_dimension_arg = [2] # subsampled input dimension for product kernel
                gamma_arg = [1]

            elif dataset == 'epileptic':
                order_hermite_arg = [10] # HP order for sum kernel
                batch_rate = [0.1]
                how_many_epochs_arg = [20]
                undersampled_rate = [1.0] # to deal with class imbalance 0.3
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [5] # HP order for prod kernel
                prod_dimension_arg = [7] # subsampled input dimension for product kernel
                gamma_arg = [0.1]

            elif dataset == 'isolet':
                order_hermite_arg = [10] # HP order for sum kernel
                batch_rate = [0.5]
                how_many_epochs_arg = [20]
                undersampled_rate = [1.] # to deal with class imbalance
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [13] # HP order for prod kernel
                prod_dimension_arg = [5] # subsampled input dimension for product kernel
                gamma_arg = [1]

            elif dataset=='intrusion':
                order_hermite_arg = [7] # HP order for sum kernel
                batch_rate = [0.01]
                how_many_epochs_arg = [200]
                undersampled_rate = [0.03] # to deal with class imbalance
                # the three below only matters for combined kernel
                order_hermite_arg_prod = [5] # HP order for prod kernel
                prod_dimension_arg = [5] # subsampled input dimension for product kernel
                gamma_arg = [1]


            #the hyperparameter setups we used for sum kernel only are in Table 4 in Section F in https://arxiv.org/pdf/2106.05042.pdf


            grid = ParameterGrid({"order_hermite": order_hermite_arg, "batch_rate": batch_rate, "n_epochs": how_many_epochs_arg, \
                                  "undersampled_rate": undersampled_rate, "order_hermite_prod": order_hermite_arg_prod, "prod_dimension": prod_dimension_arg, "gamma": gamma_arg})


        repetitions = 5 # seed: 0 to 4

        if dataset in ["credit", "census", "cervical", "adult", "isolet", "epileptic"]:

            max_aver_roc, max_aver_prc, max_roc, max_prc, max_aver_rocprc, max_elem=0, 0, 0, 0, [0,0], 0

            for elem in grid:
            # if 1:
                #print(elem, "\n")
                prc_arr_all = []; roc_arr_all = []

                for ii in range(repetitions):
                    print("\nRepetition: ",ii)


                    if grid_insidefile:
                        print(elem)
                    else:
                        print(ar)

                    # roc, prc, dir_result  = main(dataset, ii, elem["order_hermite"], elem["batch_rate"], elem["n_epochs"], elem["kernel_length"], elem["subsampled_rate"])
                    if grid_insidefile:
                        roc, prc, dir_result = main(dataset, ii, elem["order_hermite"], elem["batch_rate"], elem["n_epochs"], elem["undersampled_rate"], elem["order_hermite_prod"], elem["prod_dimension"], elem["gamma"])
                    else:
                        roc, prc, dir_result = main(dataset, ii, ar.order_hermite, ar.batch_rate, ar.epochs, ar.undersampled_rate, ar.order_hermite_prod, ar.prod_dimension, ar.gamma)



                    roc_arr_all.append(roc)
                    prc_arr_all.append(prc)


                roc_each_method_avr=np.mean(roc_arr_all, 0)
                prc_each_method_avr=np.mean(prc_arr_all, 0)
                roc_each_method_std = np.std(roc_arr_all, 0)
                prc_each_method_std = np.std(prc_arr_all, 0)
                roc_arr = np.mean(roc_arr_all, 1)
                prc_arr = np.mean(prc_arr_all, 1)

                # sys.stdout = open(dir_result+'result_txt.txt', "w")

                print("\n", "-" * 40, "\n\n")
                print("For each of the methods")
                print("Average ROC each method: ", roc_each_method_avr);
                print("Average PRC each method: ", prc_each_method_avr);
                print("Std ROC each method: ", roc_each_method_std);
                print("Std PRC each method: ", prc_each_method_std)


                print("Average over repetitions across all methods")
                print("Average ROC: ", np.mean(roc_arr)); print("Average PRC: ", np.mean(prc_arr))
                print("Std ROC: ", np.std(roc_arr)); print("Variance PRC: ", np.std(prc_arr), "\n")
                print("\n", "-" * 80, "\n\n\n")

                # sys.stdout.close()

                if np.mean(roc_arr)>max_aver_roc:
                    max_aver_roc=np.mean(roc_arr)

                if np.mean(prc_arr)>max_aver_prc:
                    max_aver_prc=np.mean(prc_arr)

                if grid_insidefile:
                    if np.mean(roc_arr) + np.mean(prc_arr)> max_aver_rocprc[0]+max_aver_rocprc[1]:
                        max_aver_rocprc = [np.mean(roc_arr), np.mean(prc_arr)]
                        max_elem = elem

            if grid_insidefile:
                print("\n\n", "*"*30, )
                print(dataset)
                print("Max ROC! ", max_aver_rocprc[0])
                print("Max PRC! ", max_aver_rocprc[1])
                if grid_insidefile:
                    print("Setup: ", max_elem)
                print('*'*100)

            filename = f"tab_results/tab_param_search_{ar.data_name}.csv"
            file = open(filename, "a+")
            file.write(
                f"{ar.data_name},{ar.epsilon},{ar.batch_rate}, {ar.order_hermite_prod},{ar.prod_dimension},{ar.order_hermite},{ar.gamma},{ar.combined_kernel},{ar.epochs},{np.std(roc_arr):.3f},{np.std(prc_arr):.3f},{np.mean(roc_arr):.3f},{np.mean(prc_arr):.3f},{int(ar.is_private)},{ar.split_eps_rate} \n")
            file.close()



        elif dataset in ["covtype", "intrusion"]: # multi-class classification problems.

            max_f1, max_aver_f1, max_elem=0, 0, 0

            for elem in grid: # uncomment this for running grid inside the file, the rest is done by boolean, grid_insidefile
                print(elem, "\n")

            # if 1:
                f1score_arr_all = []
                for ii in range(repetitions):
                    # ii = 4

                    print("\nRepetition: ",ii)

                    # f1scr = main(dataset, elem["undersampling_rates"], elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"], is_priv_arg, seed_number=ii)
                    if grid_insidefile:
                        # f1scr = main(dataset, ii, elem["order_hermite"], elem["batch_rate"], elem["n_epochs"], elem["kernel_length"], elem["subsampled_rate"])
                        f1scr = main(dataset, ii, elem["order_hermite"], elem["batch_rate"],
                                                    elem["n_epochs"], elem["undersampled_rate"],
                                                    elem["order_hermite_prod"], elem["prod_dimension"], elem["gamma"])

                    else:
                        f1scr = main(dataset, ii, ar.order_hermite, ar.batch_rate, ar.epochs,
                                                ar.undersampled_rate, ar.order_hermite_prod, ar.prod_dimension,
                                                ar.gamma)

                    f1score_arr_all.append(f1scr[0])


                f1score_each_method_avr = np.mean(f1score_arr_all, 0)
                f1score_each_method_std = np.std(f1score_arr_all, 0)
                f1score_arr = np.mean(f1score_arr_all, 1)

                print("\n", "-" * 40, "\n\n")
                print("For each of the methods")
                print("Average F1: ", f1score_each_method_avr)
                print("Std F1: ", f1score_each_method_std)


                print("Average over repetitions across all methods")
                print("Average f1 score: ", np.mean(f1score_arr))
                print("Std F1: ", np.std(f1score_arr))
                print("\n","-" * 80, "\n\n\n")

                if grid_insidefile: # for all the elements in a grid
                    if np.mean(f1score_arr)>max_aver_f1:
                        max_aver_f1=np.mean(f1score_arr)
                        max_elem = elem

            if grid_insidefile:
                print("\n\n", "Max F1! ", max_aver_f1, "*"*20)
                print("Setup: ", max_elem)
                print('*' * 30)
                
            filename = f"tab_results/tab_param_search_{ar.data_name}.csv"
            file = open(filename, "a+")
            file.write(f"{ar.data_name},{ar.epsilon},{ar.batch_rate}, {ar.order_hermite_prod},{ar.prod_dimension},{ar.order_hermite},{ar.gamma},{ar.combined_kernel},{ar.epochs},{np.std(f1score_arr):.3f},{np.mean(f1score_arr):.3f},{int(ar.is_private)},{ar.split_eps_rate} \n")
            file.close()



