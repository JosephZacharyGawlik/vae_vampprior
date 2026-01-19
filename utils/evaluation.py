from __future__ import print_function

import torch
from torch.autograd import Variable

from utils.additional_metrics import compute_active_units
from visual_evaluation import plot_images

import matplotlib.pyplot as plt

import numpy as np

import time

import os
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            data, target = Variable(data), Variable(target)

        x = data

        # calculate loss function
        loss, RE, KL = model.calculate_loss(x, average=True)

        evaluate_loss += loss.data.item()
        evaluate_re += -RE.data.item()
        evaluate_kl += KL.data.item()

        # print N digits
        if batch_idx == 1 and mode == 'validation':
            if epoch == 1:
                if not os.path.exists(dir + 'reconstruction/'):
                    os.makedirs(dir + 'reconstruction/')
                # VISUALIZATION: plot real images
                plot_images(args, data.data.cpu().numpy()[0:9], dir + 'reconstruction/', 'real', size_x=3, size_y=3)
            x_mean = model.reconstruct_x(x)
            plot_images(args, x_mean.data.cpu().numpy()[0:9], dir + 'reconstruction/', str(epoch), size_x=3, size_y=3)

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size

    if mode == 'test':
        # load all data
        # grab the test data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        test_data, test_target = [], []
        for data, lbls in data_loader:
            test_data.append(data)
            test_target.append(lbls)

        test_data, test_target = [torch.cat(test_data, 0), torch.cat(test_target, 0).squeeze()]

        # grab the train data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        full_data = []
        for data, _ in train_loader:
            full_data.append(data)

        full_data = torch.cat(full_data, 0)

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        if args.dynamic_binarization:
            full_data = torch.bernoulli(full_data)

        # print(model.means(model.idle_input))

        # VISUALIZATION: plot real images
        plot_images(args, test_data.data.cpu().numpy()[0:25], dir, 'real', size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:25])

        plot_images(args, samples.data.cpu().numpy(), dir, 'reconstructions', size_x=5, size_y=5)

        # VISUALIZATION: plot generations
        samples_rand = model.generate_x(25)

        plot_images(args, samples_rand.data.cpu().numpy(), dir, 'generations', size_x=5, size_y=5)

        if args.prior == 'vampprior':
            # VISUALIZE pseudoinputs
            pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

            plot_images(args, pseudoinputs[0:25], dir, 'pseudoinputs', size_x=5, size_y=5)

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_train = 0. # model.calculate_lower_bound(full_data, MB=args.MB)
        t_ll_e = time.time()
        print('Train lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test', S=args.S, MB=args.MB)
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0. #model.calculate_likelihood(full_data, dir, mode='train', S=args.S, MB=args.MB)) #commented because it takes too much time
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))

        # # # Compute Gaps & Active Units & Visualizations # # # # # ## # # # # # # # # # # ## # # # # # # # # # # ## # # # # # 
        # 1. ELBO Gap (Suboptimality Gap)
        # Difference between true log-likelihood and ELBO
        elbo_gap = log_likelihood_test - elbo_test
        # 2. Prior Gap
        prior_gap = evaluate_kl

        os.makedirs(dir, exist_ok=True)
        log_file = os.path.join(dir, 'test_summary_metrics.csv')

        # Extract architectural details safely
        prior_type = args.prior
        # Use getattr(object, 'attr', default) to handle models where these aren't defined
        n_pseudo = getattr(args, 'number_components', 0) if prior_type == 'vampprior' else 0
        is_weighted = getattr(args, 'weighted', False) if prior_type == 'vampprior' else False
        flow_h = getattr(args, 'flow_hidden_dim', 0) if prior_type == 'flowprior' else 0
        flow_l = getattr(args, 'flow_layers', 0) if prior_type == 'flowprior' else 0

        # Calculate Active Units
        active_units, _ = compute_active_units(model, data_loader, dir, 'cuda' if not args.no_cuda else 'cpu') 

        with open(log_file, 'w') as f:
            # Header: Added AU, RE, and BPD
            f.write('Prior_Type,PseudoInputs,Weighted,Flow_H,Flow_L,LL_Test,ELBO_Test,RE_Test,ELBO_Gap,Prior_Gap,Active_Units,BPD\n')
            
            # Calculate BPD
            bpd = -log_likelihood_test / (784 * np.log(2)) # For MNIST 28x28
            
            # Data Row
            f.write(f"{prior_type},"
                    f"{n_pseudo},"
                    f"{is_weighted},"
                    f"{flow_h},"
                    f"{flow_l},"
                    f"{log_likelihood_test:.4f},"
                    f"{elbo_test:.4f},"
                    f"{evaluate_re:.4f}," # Reconstruction error
                    f"{elbo_gap:.4f},"
                    f"{prior_gap:.4f},"
                    f"{active_units}," 
                    f"{bpd:.4f}\n")
                
        # --- 1. Gap Visualization ---
        labels = ['ELBO Gap', 'Prior Gap']
        values = [elbo_gap, prior_gap]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=['#2e6dcc', '#e74c3c'])
        plt.title(f'Gap Analysis: {prior_type}')
        plt.ylabel('Magnitude')
        plt.savefig(os.path.join(dir, 'gap_analysis.png'))
        plt.close()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
    else:
        return evaluate_loss, evaluate_re, evaluate_kl
