import argparse
import logging

import torch
import gc
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

import torch.nn.functional as F


import numpy as np
import random

from pathlib import Path




from dataset_1 import MyDataset


import resnet_1
import os

gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code.")

parser.add_argument("--network_type", "--nt", default="resnet", choices=["resnet"],
                    help="Deep network type. (default=resnet)")
parser.add_argument("--load",
                    help="Load saved network weights.")


parser.add_argument("--epochs", default=1000, type=int,
                    help="Epochs through the data. (default=2)")  
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")               
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=2)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need



def main(options):
    # Path configuration
    TRAINING_PATH = ''
    VAL_PATH = ''
    TESTING_PATH = ''
    IMG_PATH = ''

    dset_train = MyDataset(IMG_PATH, TRAINING_PATH)
    dset_val = MyDataset(IMG_PATH, VAL_PATH)
    # dset_test = MyDataset(IMG_PATH, TESTING_PATH)

    # Use argument load to distinguish training and testing
    if options.load is None:
        train_loader = DataLoader(dset_train,
                                  batch_size = options.batch_size,
                                  shuffle = True,
                                  num_workers = 4,
                                  drop_last = True
                                 )
    else:
        # Only shuffle the data when doing training
        train_loader = DataLoader(dset_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=True
                                  )


    val_loader = DataLoader(dset_val,
                             batch_size = options.batch_size,
                             shuffle = False,
                             num_workers = 4,
                             drop_last=False
                             )

    # test_loader = DataLoader(dset_test,
    #                          batch_size = options.batch_size,
    #                          shuffle = False,
    #                          num_workers = 4,
    #                          drop_last=False
    #                          )

    use_cuda = (len(options.gpuid) >= 1)

    # Training process
    if options.load is None:
        # Initial the model
        ## gen model
        if options.network_type == 'resnet':
            model = resnet_1.generate_model(50)
        
        ## cpu or gpu
        if use_cuda > 0:
            model = nn.DataParallel(model, device_ids=options.gpuid).cuda()
        else:
            model.cpu()

        print("# parameters:", sum(param.numel() for param in model.parameters()))
        for name, param in model.named_parameters():
                    print(name)
                    print(param.numel())

        # Binary cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()

        lr = options.learning_rate
        optimizer = eval("torch.optim." + options.optimizer)(model.parameters(), lr,
                                                             #momentum=options.momentum,
                                                             weight_decay=options.weight_decay)
        max_accuracy_val = 0.0
        max_accuracy_test = 0.0

        # main training loop
        for epoch_i in range(options.epochs):
            logging.info("At {0}-th epoch.".format(epoch_i))
            train_loss = 0.0
            correct_cnt = 0.0
            model.train()
            for it, train_data in enumerate(train_loader):
                data_dic = train_data

                if use_cuda:
                    imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda() 
                else:
                    imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])

                # add channel dimension: (batch_size, D, H ,W) to (batch_size, 1, D, H ,W)
                # since 3D convolution requires 5D tensors
                img_input = imgs#.unsqueeze(1)
                img_input = img_input.type(torch.cuda.FloatTensor)
                integer_encoded = labels.data.cpu().numpy()
                # target should be LongTensor in loss function
                ground_truth = Variable(torch.from_numpy(integer_encoded)).long()                                                                                             

                if use_cuda:
                    ground_truth = ground_truth.cuda()

                train_output = model(img_input)
                _, predicted = torch.max(train_output.data, 1)

                train_output = train_output.squeeze()
                
                loss = criterion(train_output, ground_truth)
                train_loss += loss.item()

                correct_cnt += (predicted == ground_truth).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_avg_loss = train_loss / (len(dset_train) / options.batch_size)
            train_avg_acu = float(correct_cnt) / len(dset_train)
            logging.info("Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss, epoch_i))
            logging.info("Average training accuracy is {0:.5f} at the end of epoch {1}".format(train_avg_acu, epoch_i))
            
            # validation -- this is a crude esitmation because there might be some paddings at the end
            val_loss = 0.0
            test_loss = 0.0
            val_correct_cnt = 0.0
            test_correct_cnt = 0.0
            model.eval()
            with torch.no_grad():
                for it, val_data in enumerate(val_loader):
                    data_dic = val_data

                    if use_cuda:
                        with torch.no_grad():
                            imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda() 
                    else:
                        with torch.no_grad():
                            imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])

                    img_input = imgs#.unsqueeze(1)
                    img_input = img_input.type(torch.FloatTensor)
                    integer_encoded = labels.data.cpu().numpy()
                    with torch.no_grad():
                        ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
                
                    if use_cuda:
                        ground_truth = ground_truth.cuda()
                    val_output = model(img_input)
                    _, predicted = torch.max(val_output.data, 1)

                
                    val_output = val_output.squeeze()
                
                    loss = criterion(val_output, ground_truth)
                    val_loss += loss.item()

                    val_correct_cnt += (predicted == ground_truth).sum().item()

                val_avg_loss = val_loss / (len(dset_val) / options.batch_size)
                val_avg_acu = float(val_correct_cnt) / len(dset_val)
                logging.info("Average validation loss is {0:.5f} at the end of epoch {1}".format(val_avg_loss, epoch_i))
                logging.info("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(val_avg_acu, epoch_i))


                if max_accuracy_val < val_avg_acu :
                    max_accuracy_val = val_avg_acu
                    torch.save(model.state_dict(), open('./ADHD' + ".nll_val_best_ADHD", 'wb'))
                print(f'Max accuracy for val: {max_accuracy_val:.5f}')


            # with torch.no_grad():
            #     for it, test_data in enumerate(test_loader):
            #         data_dic = test_data

            #         if use_cuda:
            #             with torch.no_grad():
            #                 imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda() 
            #         else:
            #             with torch.no_grad():
            #                 imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])

            #         img_input = imgs#.unsqueeze(1)
            #         img_input = img_input.type(torch.FloatTensor)
            #         integer_encoded = labels.data.cpu().numpy()
            #         with torch.no_grad():
            #             ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
                
            #         #ground_truth = ground_truth.to(torch.float)
                
            #         if use_cuda:
            #             ground_truth = ground_truth.cuda()
            #         test_output = model(img_input)
            #         #test_prob_predict = F.softmax(test_output, dim=1)
            #         #_, predict = test_prob_predict.topk(1)
            #         _, predicted = torch.max(test_output.data, 1)

                
            #         test_output = test_output.squeeze()
                
            #         if len(test_output.shape) == 1:
            #             test_output = torch.unsqueeze(test_output,0)
            #         loss = criterion(test_output, ground_truth)
            #         # loss.requires_grad_(True)
            #         test_loss += loss.item()

            #         #correct_this_batch = (predict.squeeze(1) == ground_truth).sum()
            #         #correct_cnt += (predict.squeeze(1) == ground_truth).sum()
            #         test_correct_cnt += (predicted == ground_truth).sum().item()

            #         #accuracy = float(correct_this_batch) / len(ground_truth)
            #         #logging.info("batch {0} dev loss is : {1:.5f}".format(it, loss))
            #         #logging.info("batch {0} dev accuracy is : {1:.5f}".format(it, accuracy))

            #     test_avg_loss = test_loss / (len(dset_test) / options.batch_size)
            #     test_avg_acu = float(test_correct_cnt) / len(dset_test)
            #     logging.info("Average test loss is {0:.5f} at the end of epoch {1}".format(test_avg_loss, epoch_i))
            #     logging.info("Average test accuracy is {0:.5f} at the end of epoch {1}".format(test_avg_acu, epoch_i))


            #     if max_accuracy_test < test_avg_acu :
            #         max_accuracy_test = test_avg_acu
            #         torch.save(model.state_dict(), open('./ADNI' + ".nll_test_best", 'wb'))
            #     print(f'Max accuracy for test: {max_accuracy_test:.5f}')



            #torch.save(model.state_dict(), open(options.save + ".nll_{0:.3f}.epoch_{1}".format(dev_avg_loss, epoch_i), 'wb'))

 


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
