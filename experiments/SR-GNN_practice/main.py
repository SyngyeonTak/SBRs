# Hyper Parameter Setting v
# Data set load v
# Dataset on Utilizaion
# Model train

import argparse
import pickle
import os
from utils import Data, split_validation
from model import *
import time
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'diginetica', help = 'dataset name: diginetica/yoochoose1_4/yoochoose1_64/samle')
#parser.add_argument('--method', type = str, default = 'ggnn', help = 'ggnn/gat/gcn')
parser.add_argument('--epoch', type = int, default = 1, help = 'number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=3, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty') # L2 Penalty = lambda * sum(w_i^2)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate') # w_(t+1) = w_t - eta * gradient(L(w_t))
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type = float, default = 0.1, help='split the portion of training set as validation set')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate') # new_learning_rate = old_learning_rate * lr_dc
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay') # learning_rate_t = initial_learning_rate * (lr_dc ^ floor(t / lr_dc_step))
    # step indicates batches after which it decays learning rate (step * input size / batchsize) = step * epoch

opt = parser.parse_args()

def main():

    train_data = pickle.load(open('./datasets/'+opt.dataset+'/train_sliced.txt', 'rb')) 
        # type: list; element: [(input sequences), (target item)]

    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('./datasets/'+opt.dataset+'/test_sliced.txt', 'rb'))

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=True)

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    results = []
    
    for epoch in tqdm(range(opt.epoch), desc = "Training Progress"):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        hit, mrr = train_test(model, train_data, test_data)
        flag = 0

        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1

        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch    
            flag = 1

        results.append([epoch, hit, mrr, best_result[0], best_result[1], best_epoch[0], best_epoch[1]])

        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            print('bad_counter: ', bad_counter)
            break

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

    # Save results to CSV
    with open(f'./results/SR-GNN/{opt.dataset}/training_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Recall@20', 'MMR@20', 'Best Recall@20', 'Best MMR@20', 'Best Recall@20 Epoch', 'Best MMR@20 Epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({
                'Epoch': result[0],
                'Recall@20': result[1],
                'MMR@20': result[2],
                'Best Recall@20': result[3],
                'Best MMR@20': result[4],
                'Best Recall@20 Epoch': result[5],
                'Best MMR@20 Epoch': result[6]
            })

if __name__ == '__main__':
    main()
    