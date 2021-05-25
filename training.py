#data preprocessing
import json
import numpy as np
from config import Config

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score
import random

from data.data_preprocessing import PhemeDataset
from embeddings.pre_embedding import Bert_Embedding
from BertDataset import get_randsampler
from BertDataset import bertDataset
from model.guest import GUEST


def main():
    #data preprocessing
    config = Config()
    data_prep = PhemeDataset()
    textWords_all, reactionWords_all, vLabels_all, claim_features_all, comm_features_all, tweet_struct_temp_all, text_style_all, CommAdjMats_truncate_all, comm_masks_all, text_class_all, text_id_all = data_prep.read_alldata(config.data_rumEval)
    textWords_test, reactionWords_test, vLabels_test, claim_features_test, comm_features_test, tweet_struct_temp_test, text_style_test, CommAdjMats_truncate_test, comm_masks_test, text_class_test, text_id_test = data_prep.read_alldata(config.data_rumEval_test)

    ##development dataset
    dev_data = config.data_rumEval_dev
    with open(dev_data,'r') as file_object:
                dev_idx = json.load(file_object)

    textWords = []
    reactionWords = []
    vLabels = []
    claim_features = []
    comm_features = [] 
    tweet_struct_temp = [] 
    text_style = [] 
    CommAdjMats_truncate = [] 
    comm_masks = [] 
    text_class = []

    textWords_dev = []
    reactionWords_dev = [] 
    vLabels_dev = []
    claim_features_dev = []
    comm_features_dev = [] 
    tweet_struct_temp_dev = [] 
    text_style_dev = [] 
    CommAdjMats_truncate_dev = [] 
    comm_masks_dev = [] 
    text_class_dev = []


    for i in range(len(vLabels_all)):
        if text_id_all[i] in dev_idx:
            
            textWords_dev.append(textWords_all[i])
            reactionWords_dev.append(reactionWords_all[i])
            vLabels_dev.append(vLabels_all[i])
            claim_features_dev.append(claim_features_all[i])
            comm_features_dev.append(comm_features_all[i])
            tweet_struct_temp_dev.append(tweet_struct_temp_all[i])
            text_style_dev.append(text_style_all[i])
            CommAdjMats_truncate_dev.append(CommAdjMats_truncate_all[i])
            comm_masks_dev.append(comm_masks_all[i])
            text_class_dev.append(text_class_all[i])
        else:
            textWords.append(textWords_all[i])
            reactionWords.append(reactionWords_all[i])
            vLabels.append(vLabels_all[i])
            claim_features.append(claim_features_all[i])
            comm_features.append(comm_features_all[i])
            tweet_struct_temp.append(tweet_struct_temp_all[i])
            text_style.append(text_style_all[i])
            CommAdjMats_truncate.append(CommAdjMats_truncate_all[i])
            comm_masks.append(comm_masks_all[i])
            text_class.append(text_class_all[i])
            
    ##del temporal feature
    tweet_struct_temp = data_prep.get_struct_new(tweet_struct_temp)
    tweet_struct_temp_dev = data_prep.get_struct_new(tweet_struct_temp_dev)
    tweet_struct_temp_test = data_prep.get_struct_new(tweet_struct_temp_test)

    ##Comm Features truncate
    comm_features_trunc = data_prep.get_comm_fea_truncs(comm_features,  config.nums_comm)
    comm_features_trunc_dev = data_prep.get_comm_fea_truncs(comm_features_dev,  config.nums_comm)
    comm_features_trunc_test = data_prep.get_comm_fea_truncs(comm_features_test,  config.nums_comm)

    ##Bert Embedding
    Bert_Embed = Bert_Embedding(config)

    claim_embeds, claim_comm_embeds, words = Bert_Embed.Pre_Bert_Embedding(textWords, reactionWords, Bert_Embed.model_preBert)
    claim_embeds_dev, claim_comm_embeds_dev, words_dev = Bert_Embed.Pre_Bert_Embedding(textWords_dev, reactionWords_dev, Bert_Embed.model_preBert)
    claim_embeds_test, claim_comm_embeds_test, words_test = Bert_Embed.Pre_Bert_Embedding(textWords_test, reactionWords_test, Bert_Embed.model_preBert)

    ##number of comments
    claim_comm_embeds_trunc_train, claim_embeds_train = Bert_Embed.Embed_truncate(claim_embeds, claim_comm_embeds, config.nums_comm)
    claim_comm_embeds_trunc_dev, claim_embeds_dev = Bert_Embed.Embed_truncate(claim_embeds_dev, claim_comm_embeds_dev, config.nums_comm)
    claim_comm_embeds_trunc_test, claim_embeds_test = Bert_Embed.Embed_truncate(claim_embeds_test, claim_comm_embeds_test, config.nums_comm)

    ##data set for negative log loss
    RandomSampler = get_randsampler(vLabels)

    #training
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    train_acc_plot = []
    train_loss_plot = []
    train_epoch = []
    dev_acc_plot = []
    dev_loss_plot = []
    dev_epoch = []
    test_acc_plot = []
    test_loss_plot = []
    test_epoch = []

    def correct_prediction(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct

    def genMetrics(output, labels):
        preds = np.argmax(output, axis=1)
        print("preds ", preds)
        print("labels", labels)
        recall = recall_score(labels, preds, average='macro')
        macroprec = precision_score(labels, preds, average='macro')
        macrof1 = f1_score(labels, preds, average='macro') 
        return recall, macroprec, macrof1
        
    model = GUEST(nfeat=config.nums_bert, bi_num_hidden=config.bi_num_hidden,
                nums_feat = config.nums_feat, nembd_1=config.nembd_1, nembd_2=config.nembd_2,
                nins=config.nums_comm, nclass=config.nclass, nlayer=config.nlayer, pool="att")  #nlayer=3
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sheduler = StepLR(optimizer, config.step, config.gamma)

    train_data = bertDataset(claim_comm_embeds_trunc_train, vLabels, claim_embeds_train, claim_features, 
                            comm_features_trunc, comm_masks, tweet_struct_temp) 
    train_dataloader = DataLoader(train_data, config.batch_size, sampler=RandomSampler)

    dev_data = bertDataset(claim_comm_embeds_trunc_dev, vLabels_dev, claim_embeds_dev, claim_features_dev, 
                            comm_features_trunc_dev, comm_masks_dev, tweet_struct_temp_dev) 
    dev_dataloader = DataLoader(dev_data, batch_size=len(dev_data))



    test_data = bertDataset(claim_comm_embeds_trunc_test, vLabels_test, claim_embeds_test, claim_features_test, 
                            comm_features_trunc_test, comm_masks_test, tweet_struct_temp_test) 
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

        
    for epoch in range(config.epoch):
        print('epoch', epoch)
        model.train()
        running_loss = 0.0
        correct_pred = 0.0
        for index, data in enumerate(train_dataloader):
            comm_batch, label_batch, claim_batch, claim_fea_batch, comm_fea_batch, masks_batch, tweet_struct_batch = data
            comm_batch = comm_batch.cuda()
            label_batch = label_batch.cuda()
            claim_batch = claim_batch.cuda()
            claim_fea_batch = claim_fea_batch.cuda()
            comm_fea_batch = comm_fea_batch.cuda()
            masks_batch =  masks_batch.cuda()
            tweet_struct_batch = tweet_struct_batch.cuda()
            optimizer.zero_grad()
            outputs,_ ,_ ,_ = model(comm_batch, claim_batch, claim_fea_batch, comm_fea_batch, masks_batch, tweet_struct_batch)
            #loss = F.nll_loss(outputs, label_batch)
            loss = F.cross_entropy(outputs, label_batch)
            correct_pred += correct_prediction(outputs, label_batch)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            print('Acc: %lf, Loss: %lf' % (correct_pred / ((index + 1)*32), running_loss / (index+1)))
        train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct_pred / len(train_dataloader.dataset)
        print('Train total acc: %lf, total loss: %lf\r\n' % (train_accuracy, train_loss))

        model.eval()
        running_loss_dev = 0.0
        correct_pred_dev = 0.0
        running_loss_test = 0.0
        correct_pred_test = 0.0
        with torch.no_grad():
            
            #dev data
            for index, data in enumerate(dev_dataloader):
                comm_batch, label_batch, claim_batch, claim_fea_batch, comm_fea_batch, masks_batch, tweet_struct_batch = data
                comm_batch = comm_batch.cuda()
                label_batch = label_batch.cuda()
                claim_batch = claim_batch.cuda()
                claim_fea_batch = claim_fea_batch.cuda()
                comm_fea_batch = comm_fea_batch.cuda()
                masks_batch =  masks_batch.cuda()
                tweet_struct_batch = tweet_struct_batch.cuda()

                outputs,_ ,_ ,_ = model(comm_batch, claim_batch, claim_fea_batch, comm_fea_batch, masks_batch, tweet_struct_batch)
                loss = F.nll_loss(outputs, label_batch)

                correct_pred_dev += correct_prediction(outputs, label_batch)
                metrics = genMetrics(outputs.cpu().data.numpy(), label_batch.cpu().data.numpy())
                print("metrics", metrics)
                running_loss_dev += loss.item()
                print('Acc_dev: %lf, Loss_test: %lf' % (correct_pred_dev / ((index + 1)*len(dev_dataloader.dataset)), running_loss_dev / (index + 1)))
                print('recall: %lf, microprec: %lf, microf1: %lf' % (metrics[0], metrics[1], metrics[2]))

            dev_loss = running_loss_dev / len(dev_dataloader)
            dev_accuracy = correct_pred_dev / len(dev_dataloader.dataset)
            print('dev total acc: %lf, total dev loss: %lf\r\n' % (dev_accuracy, dev_loss))
            
            for index, data in enumerate(test_dataloader):
                comm_batch, label_batch, claim_batch, claim_fea_batch, comm_fea_batch, masks_batch, tweet_struct_batch = data
                comm_batch = comm_batch.cuda()
                label_batch = label_batch.cuda()
                claim_batch = claim_batch.cuda()
                claim_fea_batch = claim_fea_batch.cuda()
                comm_fea_batch = comm_fea_batch.cuda()
                masks_batch =  masks_batch.cuda()
                tweet_struct_batch = tweet_struct_batch.cuda()
                outputs,_ ,_ ,_ = model(comm_batch, claim_batch, claim_fea_batch, comm_fea_batch, masks_batch, tweet_struct_batch)
                loss = F.nll_loss(outputs, label_batch)

                correct_pred_test += correct_prediction(outputs, label_batch)
                metrics = genMetrics(outputs.cpu().data.numpy(), label_batch.cpu().data.numpy())
                print("metrics", metrics)
                running_loss_test += loss.item()
                print('Acc_test: %lf, Loss_test: %lf' % (correct_pred_test / ((index + 1)*len(test_dataloader.dataset)), running_loss_test / (index + 1)))
                print('recall: %lf, microprec: %lf, microf1: %lf' % (metrics[0], metrics[1], metrics[2]))

        test_loss = running_loss_test / len(test_dataloader)
        test_accuracy = correct_pred_test / len(test_dataloader.dataset)
        print('Test total acc: %lf, total Test loss: %lf\r\n' % (test_accuracy, test_loss))

        train_epoch.append(epoch)
        train_loss_plot.append(train_loss)
        train_acc_plot.append(train_accuracy)
        
        dev_epoch.append(epoch)
        dev_loss_plot.append(dev_loss)
        dev_acc_plot.append(dev_accuracy)
        

        test_epoch.append(epoch)
        test_loss_plot.append(test_loss)
        test_acc_plot.append(test_accuracy)


        if epoch%20 == 0:
            plt.figure()
            plt.plot(train_epoch,train_loss_plot, label='train loss')
            plt.ylim((0, 1.5))
            plt.yticks(np.linspace(0,1.5,20))
            plt.grid(axis="y")  
            plt.legend(loc='upper left')
            plt.show()

            plt.figure()
            plt.plot(train_epoch,train_acc_plot, label='train accuracy')
            plt.ylim((0, 1))
            plt.yticks(np.linspace(0,1,20))
            plt.grid(axis="y")
            plt.legend(loc='upper left')
            plt.show()

            
            plt.figure()
            plt.plot(dev_epoch,dev_loss_plot, label='dev loss')
            plt.ylim((0, 1.5))
            plt.yticks(np.linspace(0,1.5,20))
            plt.grid(axis="y")  
            plt.legend(loc='upper left')
            plt.show()

            plt.figure()
            plt.plot(train_epoch,dev_acc_plot, label='dev accuracy')
            plt.ylim((0, 1))
            plt.yticks(np.linspace(0,1,20))
            plt.grid(axis="y")
            plt.legend(loc='upper left')
            plt.show()
            

            plt.figure()
            plt.plot(test_epoch,test_loss_plot, label='test loss')
            plt.ylim((0, 3))
            plt.yticks(np.linspace(0,3,20))
            plt.grid(axis="y")  
            plt.legend(loc='upper left')
            plt.show()

            plt.figure()
            plt.plot(test_epoch,test_acc_plot, label='test accuracy')
            plt.ylim((0, 1))
            plt.yticks(np.linspace(0,1,21))
            plt.grid(axis="y")
            plt.legend(loc='upper left')
            plt.show()

if __name__ == "__main__":
    main()
      