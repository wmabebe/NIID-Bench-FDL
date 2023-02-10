import os
import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model import *
from datasets import MNIST_truncated, CIFAR10_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData, CIFAR100_truncated, ImageFolder_custom
from math import sqrt

import torch.nn as nn

import torch.optim as optim
import torchvision.utils as vutils
import time
import random

from models.mnist_model import Generator, Discriminator, DHead, QHead
from config import params
import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file

import random
import networkx as nx

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    #np.random.seed(2020)
    #torch.manual_seed(2020)

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
        print("y_train",y_train.shape)
        print("y_test",y_test.shape)

    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 ==1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1>0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0,3999,4000,dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)
    
    #elif dataset == 'covtype':
    #    cov_type = sk.fetch_covtype('./data')
    #    num_train = int(581012 * 0.75)
    #    idxs = np.random.permutation(581012)
    #    X_train = np.array(cov_type['data'][idxs[:num_train]], dtype=np.float32)
    #    y_train = np.array(cov_type['target'][idxs[:num_train]], dtype=np.int32) - 1
    #    X_test = np.array(cov_type['data'][idxs[num_train:]], dtype=np.float32)
    #    y_test = np.array(cov_type['target'][idxs[num_train:]], dtype=np.int32) - 1
    #    mkdirs("data/generated/")
    #    np.save("data/generated/X_train.npy",X_train)
    #    np.save("data/generated/X_test.npy",X_test)
    #    np.save("data/generated/y_train.npy",y_train)
    #    np.save("data/generated/y_test.npy",y_test)

    elif dataset in ('rcv1', 'SUSY', 'covtype'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_train = X_train.todense()
        num_train = int(X_train.shape[0] * 0.75)
        if dataset == 'covtype':
            y_train = y_train-1
        else:
            y_train = (y_train+1)/2
        idxs = np.random.permutation(X_train.shape[0])

        X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
        y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
        X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
        y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)

    elif dataset in ('a9a'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_test, y_test = load_svmlight_file("../../../data/{}.t".format(dataset))
        X_train = X_train.todense()
        X_test = X_test.todense()
        X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = (y_train+1)/2
        y_test = (y_test+1)/2
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)


    n_train = y_train.shape[0]
    
    if partition == "manual":
        label_idxs = {i:[] for i in np.unique(y_train)}
        label_node_map = {i:[] for i in label_idxs.keys()} 
        label_node_map[0] = [0,5,10,15,20]
        label_node_map[1] = [0,5,10,15,20]
        label_node_map[2] = [1,6,11,16,21]
        label_node_map[3] = [1,6,11,16,21]
        label_node_map[4] = [2,7,12,17,22]
        label_node_map[5] = [2,7,12,17,22]
        label_node_map[6] = [3,8,13,18,23]
        label_node_map[7] = [3,8,13,18,23]
        label_node_map[8] = [4,9,14,19]
        label_node_map[9] = [4,9,14,19]

        
        for i,label in enumerate(y_train):
            label_idxs[label].append(i)
        
        net_dataidx_map = {i:[] for i in range(n_parties)}
        for label, idxs in label_idxs.items():
            batch_idxs = np.array_split(idxs, len(label_node_map[label]))
            for i, net_id in enumerate(label_node_map[label]):
                net_dataidx_map[net_id] += list(batch_idxs[i])
        

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
       


    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
        elif dataset in ('tinyimagenet'):
            K = 200
            # min_require_size = 100
        elif dataset in ('cifar100'):
            K = 100

        N = y_train.shape[0]
        #np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1


    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        
    elif partition == "mixed":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        times=[1 for i in range(10)]
        contain=[]
        for i in range(n_parties):
            current=[i%K]
            j=1
            while (j<2):
                ind=random.randint(0,K-1)
                if (ind not in current and times[ind]<2):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        

        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*n_train)

        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)

            proportions_k = np.random.dirichlet(np.repeat(beta, 2))
            #proportions_k = np.ndarray(0,dtype=np.float64)
            #for j in range(n_parties):
            #    if i in contain[j]:
            #        proportions_k=np.append(proportions_k ,proportions[j])

            proportions_k = (np.cumsum(proportions_k)*len(idx_k)).astype(int)[:-1]

            split = np.split(idx_k, proportions_k)
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist=list(trainable)
    N=0
    for params in paramlist:
        N+=params.numel()
        # logger.info("params.data:", params.data)
    X=torch.empty(N,dtype=torch.float64)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel
    # logger.info("get trainable x:", X)
    return X


def put_trainable_parameters(net,X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_val_dataloader(dataset, datadir, datasize, val_bs):
    val_dl = None
    if dataset == 'tinyimagenet':
        random_ids = np.random.randint(100000, size=datasize)
        val_indices = random_ids

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        val_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(datadir,
                                                transform=transforms.Compose([
                                                transforms.Resize(32), 
                                                transforms.ToTensor(),
                                                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                                                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])),
            #Phuong 09/26 drop_last=False -> True
            batch_size=val_bs, drop_last=True, sampler=SubsetRandomSampler(val_indices))
    
    elif dataset == 'fmnist':
        dl_obj = FashionMNIST_truncated
        transform_val = transforms.Compose([
                transforms.ToTensor(),])
        
        random_ids = np.random.randint(10000, size=datasize)
        val_indices = random_ids

        val_ds = dl_obj(datadir, dataidxs=val_indices, train=True, transform=transform_val, download=True)
        val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=val_bs, shuffle=True, drop_last=False)
    
    elif dataset == "cifar10":
        dl_obj = CIFAR10_truncated
        transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

            ])
        random_ids = np.random.randint(10000, size=datasize)
        val_indices = random_ids

        val_ds = dl_obj(datadir, dataidxs=val_indices, train=True, transform=transform_val, download=True)
        val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=val_bs, shuffle=True, drop_last=False)


    return val_dl



def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10','cifar100', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY','tinyimagenet'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'femnist':
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])


        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
        
        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif dataset == 'tinyimagenet':        
            # random_ids = np.random.randint(1000, size=datasize)
            # train_indices = random_ids

            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]

            train_dl = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(datadir +"/train",
                                                transform=transforms.Compose([
                                                    transforms.Resize(32), 
                                                    transforms.ToTensor(),
                                                    # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                                                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])),
                #Phuong 09/26 drop_last=False -> True
                batch_size=train_bs, drop_last=True)
            
            test_dl = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(datadir +"/test",
                                                transform=transforms.Compose([
                                                    transforms.Resize(32), 
                                                    transforms.ToTensor(),
                                                    # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                                                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])),
                #Phuong 09/26 drop_last=False -> True
                batch_size=test_bs, drop_last=True)

            return train_dl, test_dl, None, None


        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last= dataset in ['cifar100'])
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx

#Flatten all layers of a gradient
def flatten_layers(gradient):
    X_flat = np.array([])
    for name,layer in sorted(gradient.items()):
        cur = np.array(layer,dtype=np.float64).flatten()
        X_flat = np.concatenate((X_flat,cur),axis=0)
    return X_flat

#Calculate the radians between two gradients
def get_radians(g1,g2):
    unit_vector_1 = g1 / np.linalg.norm(g1) if np.linalg.norm(g1) != 0 else 0
    unit_vector_2 = g2 / np.linalg.norm(g2) if np.linalg.norm(g2) != 0 else 0
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    radians = np.arccos(dot_product)
    return radians

#BFTM methods
def show(matrix):
    print("",end="  ")
    for k in sorted(matrix.keys()):
        print(k,end="  ")
    
    for i,row in sorted(matrix.items()):
        print("\n" + str(i),end="  ")
        for j in row:
            print(matrix[i][j],end="  ")
    print()
        

def size(matrix):
    size = 0
    for _,row in matrix.items():
        #size += len([r for r in row.values() if r != None])
        for _,v in row.items():
            #print("V:",v)
            size += 1 if v != None else 0
    return size

def is_complete(matrix,size):
    for _,row in matrix.items():
        if len(row) < size:
            return False
    return True

def exists(matrix,i,j):
    return j in matrix[i].keys()

def intersection(matrix,n1,n2):
    inter = []
    for n,v in matrix[n1.id].items():
        if v == None and matrix[n2.id][n] == None:
            inter.append(n)
    return inter

def add_entry(matrix,i,j,replace=False):
    if j not in matrix[i].keys():
        matrix[i][j] = abs(i - j)
    else:
        if replace:
            matrix[i][j] = abs(i - j)

def get_preds(model, dataloader, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    pred_labels_list = np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, out.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, out.cpu().numpy())

    if was_training:
        model.train()

    return torch.from_numpy(pred_labels_list.reshape(-1,10))


def kl_divergence(id1,id2,m1,m2,cache_val_preds,val_dl,device="cpu"):
    if m1 == m2:
        return 0
    
    m1_preds = get_preds(m1,val_dl,device) if cache_val_preds[id1] == None else cache_val_preds[id1]
    m2_preds = get_preds(m2,val_dl,device) if cache_val_preds[id2] == None else cache_val_preds[id2]

    cache_val_preds[id1] = m1_preds
    cache_val_preds[id2] = m2_preds

    print("PRED SHAPE:", m1_preds.shape)

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    result = kl_loss(m1_preds, m2_preds)
    print("\t",result)

    return result

def get_signed_radians(grad1,grad2):
    g1 = flatten_layers(grad1)
    g2 = flatten_layers(grad2)
    if np.array_equal(g1, g2):
        return 0
    angle = -1 if g1[0]*g2[1] - g1[1]*g2[0] < 0 else 1
    unit_vector_1 = g1 / np.linalg.norm(g1) if np.linalg.norm(g1) != 0 else 0
    unit_vector_2 = g2 / np.linalg.norm(g2) if np.linalg.norm(g2) != 0 else 0
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    radians = np.arccos(dot_product)
    if isinstance(radians,list):
        print("BUGGY angle: ",radians)
    return radians * angle


def update_matrix(G,M,C,cache_val_preds,adj_list,sim="grad",val_dl=None, device="cpu"):
    for node,adj in G.adj.items():
        for _1_hop,_ in adj.items():
            if sim == "grad":
                M[node.id][_1_hop.id] = get_signed_radians(node.model.grads,_1_hop.model.grads) #node.model - _1_hop.model
                M[_1_hop.id][node.id] = get_signed_radians(_1_hop.model.grads,node.model.grads) #_1_hop.model - node.model
            elif sim == "kl":
                print(node.id, "-", _1_hop.id)
                M[node.id][_1_hop.id] = kl_divergence(node.id,_1_hop.id,node.model,_1_hop.model,cache_val_preds,val_dl,device) #node.model - _1_hop.model
                M[_1_hop.id][node.id] = kl_divergence(_1_hop.id,node.id,_1_hop.model,node.model,cache_val_preds,val_dl,device) #_1_hop.model - node.model
            C[node.id].add(_1_hop.id)
            C[_1_hop.id].add(node.id)
    for node in list(G.nodes):
        for n_id,cache in C.items():
            if node.id in cache and node.id != n_id:
                for _2_hop in cache:
                    if sim == "grad":
                        M[node.id][_2_hop] = get_signed_radians(node.model.grads,adj_list[_2_hop].model.grads) #node.model - adj_list[_2_hop].model
                        M[_2_hop][node.id] = get_signed_radians(adj_list[_2_hop].model.grads,node.model.grads) #adj_list[_2_hop].model - node.model
                    elif sim == "kl":
                        print(node.id, "-", _2_hop)
                        M[node.id][_2_hop] = kl_divergence(node.id,_2_hop,node.model,adj_list[_2_hop].model,cache_val_preds,val_dl,device) #node.model - adj_list[_2_hop].model
                        M[_2_hop][node.id] = kl_divergence(_2_hop,node.id,adj_list[_2_hop].model,node.model,cache_val_preds,val_dl,device) #adj_list[_2_hop].model - node.model
            

def BFTM(G,M,C,degree):
    #show(M)
    subset = [n for n in list(G.nodes) if None in M[n.id].values()]
    print("Subset size: ",len(subset))
    #print("SUB: ",[n.id for n in subset])
    if not subset:
        return G
    GN = nx.Graph()
    done = [n for n in list(G.nodes) if None not in M[n.id].values()]
    #print("DONE: ",[n.id for n in done])
    queue = [random.choice(subset)]
    #print("root:",queue[0].id)
    temp = {n:[] for n in subset}
    while len(queue) > 0 and len(done) < len(list(G.nodes)):
        node = queue.pop(0)
        GN.add_node(node)
        candidates = [n for n in subset if M[node.id][n.id] == None and node not in temp[n]]

        #ranks = {c:set.intersection(C[node.id],C[c.id]) for c in candidates}
        #candidates = sorted(candidates, key=lambda x: ranks[x], reverse=False)


        size = max(degree - len(temp[node]),0)
        size = min(len(candidates),size)
        next_neigbors = random.sample(candidates,size)

        #next_neigbors = candidates[:size]
        
        

        #print(node.id,"-->",str([c.id for c in candidates]),"-->",str([c.id for c in next_neigbors]))
        temp[node] += next_neigbors
        #Add new edges
        for n in next_neigbors:
            GN.add_edge(node,n)
            temp[n].append(node)
            temp[node].append(n)
            if n not in queue:
                queue.append(n)
        done.append(node)
        subset.remove(node)
        if len(queue) == 0 and len(subset) > 0:
            queue = [random.choice(subset)]
    return GN

def is_empty(clusters):
    for _,cluster in clusters.items():
        if len(cluster) > 0:
            return False
    return True

def greedy_cliques(cliques,clusters,matrix):
    if len(clusters) == 0:
        return cliques
    cid = random.choice(range(len(clusters)))
    cluster0 = clusters.pop(cid)
    pid = random.choice(range(len(cluster0)))
    p0 = cluster0.pop(pid)
    clique = [p0]
    
    for cid,cluster in enumerate(clusters):
        dist = float('-Inf')
        idx = 0
        for i,p in enumerate(cluster):
            d = sum([matrix[p.id][c.id] for c in clique])
            if d > dist:
                dist = d
                idx = i
        item = cluster.pop(idx)
        clique.append(item)
        if len(cluster) <= 0:
            clusters.pop(cid)
    if len(cluster0) > 0:
        clusters.append(cluster0)
    cliques.append(clique)
    return greedy_cliques(cliques,clusters,matrix)

def pcc_clique(clusters,strategy, labels, cut=0, matrix=None):
    G = nx.Graph()
    cliques = []
    size = len(clusters)
    if strategy == "rand":
        for _,cluster in clusters.items():
            clique_size = len(cluster) if len(cluster) < size else size
            cliques.append( random.sample(cluster,clique_size))
    
    elif strategy == "optim":
        while len(cliques) < size:
            clique = []
            for _,cluster in clusters.items():
                if len(cluster):
                    element = random.choice(cluster)
                    clique.append(element)
                    del cluster[cluster.index(element)]
            if clique:
                cliques.append(clique)
    
    elif strategy == "greedy":
        cliques = greedy_cliques([],list(clusters.values()),matrix)[:len(clusters)]
    
    for clique in cliques:
        if len(clique) == 1:
            G.add_node(clique[0])
            continue
        for n1 in clique:
            for n2 in clique:
                if n1 != n2:
                    G.add_edge(n1,n2)

    print("pcc_clique CUT:", cut)
    cliques_on_ring(cliques,labels,G,cut)
    return G


def sampled_clique(clusters,strategy):
    G = nx.Graph()
    sample = []
    #Sample 'size' nodes from a single cluster
    if strategy == "rand":
        size = len(clusters)
        while len(sample) < size:
            cluster = random.choice(clusters)
            if len(cluster) >= size:
                sample = random.sample(cluster,size)
    #Sample 1 choice from each cluster
    elif strategy == "optim":
        for _,cluster in clusters.items():
            if len(cluster) > 0:
                sample.append(random.choice(cluster))
    for n1 in sample:
        for n2 in sample:
            if n1 != n2:
                G.add_edge(n1,n2)
    return G
    
        
def clique_the_cliques(cliques,labels,G):
    #Clique the cliques
    for idx1,clique1 in enumerate(cliques):
        for idx2,clique2 in enumerate(cliques):
            if idx1 < idx2:
                n1 = random.choice(clique1)
                candidates = [n for n in clique2 if labels[n.id] != labels[n1.id]]
                if candidates:
                    n2 = random.choice(candidates)
                    G.add_edge(n1,n2)
                else:
                    break

def get_partitions(cliques,cut=1):
    cliques.sort(key=len)
    k, m = divmod(len(cliques), cut)
    return list(cliques[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(cut))

def cliques_on_ring(cliques,labels,G, cut=0):
    print("cliques_on_ring CUT:", cut)
    #if partitions
    if cut > 1:
        partitions = get_partitions(cliques,cut)
        for partition in partitions:
            if len(partition) > 1:
                n1 = random.choice(partition[0])
                for idx,clique in enumerate(partition):
                    if idx > 0:# and idx not in cuts:
                        candidates = [n for n in clique if labels[n.id] != labels[n1.id]]
                        candidates = candidates if len(candidates) > 0 else clique
                        n2 = random.choice(candidates)
                        G.add_edge(n1,n2)
                    n1 = random.choice(clique)

    else:
        n1 = random.choice(cliques[0]) 
        for idx,clique in enumerate(cliques):
            if idx > 0:# and idx not in cuts:
                candidates = [n for n in clique if labels[n.id] != labels[n1.id]]
                candidates = candidates if len(candidates) > 0 else clique
                n2 = random.choice(candidates)
                G.add_edge(n1,n2)
            n1 = random.choice(clique)
        
        #Attach head and tail cliques
        candidates = [n for n in cliques[0] if labels[n.id] != labels[n1.id]]
        candidates = candidates if len(candidates) > 0 else cliques[0]
        n2 = random.choice(candidates)
        G.add_edge(n1,n2)

def manual_cliques(adj_list,cliques):
    G = nx.Graph()
    for clique in cliques:
        for n1 in clique:
            for n2 in clique:
                if n1 != n2:
                    G.add_edge(adj_list[n1],adj_list[n2])
    
    n1 = cliques[0][0]
    for idx,clique in enumerate(cliques):
        if idx > 0:
            n2 = clique[2]
            G.add_edge(adj_list[n1],adj_list[n2])
            n1 = clique[0]
        
    #Attach head and tail cliques
    n1 = cliques[0][3]
    n2 = cliques[len(cliques) - 1][1]
    G.add_edge(adj_list[n1],adj_list[n2])
    return G

def m_cliques(adj_list,labels,matrix,topology="clique",cut=0):
    num_clusters = list(np.unique(labels))
    clusters = {i:[] for i in num_clusters}
    hood = {n.id:[i for i in num_clusters if i != labels[n.id]] for n in adj_list}
    print("m_cliques CUT:", cut)
    #Add nodes to clusters
    for idx,n in enumerate(adj_list):
        clusters[labels[idx]].append(n)
    
    if topology == "sample_rand":
        G_cliques = sampled_clique(clusters,"rand")
        print("Rand sampled clique size:", G_cliques.number_of_nodes())
    elif topology == "sample_optim":
        G_cliques = sampled_clique(clusters,"optim")
        print("Optim sampled clique size:", G_cliques.number_of_nodes())
    elif topology == "pcc_rand":
        G_cliques = pcc_clique(clusters,"rand",labels,cut)
        print("Rand sampled clique size:", G_cliques.number_of_nodes())
    elif topology == "pcc_optim":
        G_cliques = pcc_clique(clusters,"optim",labels,cut)
        print("Optim sampled clique size:", G_cliques.number_of_nodes())
    elif topology == "pcc_greedy":
        G_cliques = pcc_clique(clusters,"greedy",labels,cut,matrix)
        print("Greedy sampled clique size:", G_cliques.number_of_nodes())
    else:
        G_cliques = nx.Graph()
        cliques = []
        while not is_empty(clusters):
            clique = []
            
            #Add clique nodes
            for _,cluster in clusters.items():
                if len(cluster) > 0:
                    n = random.choice(cluster)
                    clique.append(n)
                    cluster.remove(n)
            
            #Cliqify
            for n1 in clique:
                for n2 in clique:
                    if n1 != n2:
                        G_cliques.add_edge(n1,n2)
            
            #Aggregate clique
            cliques.append(clique)

        if topology == "clique":
            clique_the_cliques(cliques,labels,G_cliques)
        elif topology == "ring":
            cliques_on_ring(cliques,labels,G_cliques)
    
    return G_cliques
    
def BFTM_(adj_list,labels):
    G_prime = nx.Graph()
    num_clusters = list(np.unique(labels))
    clusters = {i:[] for i in num_clusters}
    hood = {n.id:[i for i in num_clusters if i != labels[n.id]] for n in adj_list}
    
    #Add nodes to clusters
    for idx,n in enumerate(adj_list):
        clusters[labels[idx]].append(n.id)
    
    root_cluster = random.choice(num_clusters)
    root_id = random.choice(list(clusters[root_cluster]))
    queue = [adj_list[root_id]]
    clusters[labels[root_id]].remove(root_id)
        
        
    #BFTM
    while len(queue) > 0:
        node = queue.pop(0)
        for c_id in hood[node.id]:
            if len(clusters[c_id]) > 0:
                sample_id = random.choice(clusters[c_id])
                clusters[labels[sample_id]].remove(sample_id)
                queue.append(adj_list[sample_id])
                hood[sample_id].remove(labels[node.id])
                G_prime.add_edge(node,adj_list[sample_id])
        hood[node.id] = None
        #Handle leftover nodes
        if len(queue) == 0:
            remaining = [c for i,c in clusters.items() if len(c) > 0]
            for rem_cluster in remaining:
                for n in rem_cluster:
                    added = False
                    while not added:
                        rand_n = random.choice(list(G_prime.nodes))
                        if labels[rand_n.id] != labels[n.id]:
                            G_prime.add_edge(n,rand_n)
                            added = True
        
    
    #Cliqify
    for node in list(G_prime.nodes):
        if G_prime.degree(node) < len(num_clusters) - 1:
            for _1_hop in list(G_prime.neighbors(node)):
                for _2_hop in list(G_prime.neighbors(_1_hop)):
                    if _2_hop != node and G_prime.degree(_2_hop) < len(num_clusters) - 1:
                        G_prime.add_edge(node,_2_hop)
    
    return G_prime