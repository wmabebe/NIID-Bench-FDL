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
import copy


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


from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def download_and_unzip(url, extract_to='.'):
    """Download and unzip a file.

    Args:
        url (string): URL of the zip file.
        extract_to (str, optional): Output directory. Defaults to '.'.
    """
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

#Create a directory given a path
def mkdirs(dirpath):
    """Create a directory.

    Args:
        dirpath (string): The path of the directory to be created.
    """
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):
    """Load MNIST data.

    Args:
        datadir (string): MNIST data directory.

    Returns:
        tuple: Train and test data and labels.
    """

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
    """Load FMNIST data.

    Args:
        datadir (string): FMNIST data directory.

    Returns:
        tuple: Train and test data and labels.
    """

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
    """Load SVHN data.

    Args:
        datadir (string): SVHN data directory.

    Returns:
        tuple: Train and test data and labels.
    """

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
    """Load CIFAR-10 data.

    Args:
        datadir (string): CIFAR-10 data directory.

    Returns:
        tuple: Train and test data and labels.
    """

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir):
    """Load CELEBA data.

    Args:
        datadir (string): CELEBA data directory.

    Returns:
        tuple: Train and test data and labels.
    """

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
    """Load FEMNIST data.

    Args:
        datadir (string): FEMNIST data directory.

    Returns:
        tuple: Train and test data and labels.
    """

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
    """Load CIFAR-100 data.

    Args:
        datadir (string): CIFAR-100 data directory.

    Returns:
        tuple: Train and test data and labels.
    """

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    """Load TinyImagenet data.

    Args:
        datadir (string): TinyImagenet data directory.

    Returns:
        tuple: Train and test data and labels.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    """_summary_

    Args:
        y_train (nparray): Train labels.
        net_dataidx_map (dict): Dictionary containing client:data-indices mapping.
        logdir (string): Output directory.

    Returns:
        dict: Dictionary containing client:class-counts mapping
    """

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    """This method partitions the dataset and divides it amongst the clients.

    Args:
        dataset (string): Dataset name.
        datadir (string): Directory where the dataset is located.
        logdir (string): Output directory.
        partition (string): How to partition the dataset.
        n_parties (int): Number of participating clients.
        beta (float, optional): Concentration parameter for distribution based labe skew. Defaults to 0.4.

    Returns:
        tuple: Containing train and test data as well as client:data-indices 
        dictionary and client:class-counts dictionary.
    """
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
    """Return trainable parameter values as a vector (only the first parameter set).

    Args:
        net (torch.nn): Pytorch model.

    Returns:
        torch.array: Trainable network values.
    """
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
    """Replace trainable parameter values by the given vector (only the first parameter set).

    Args:
        net (torch.nn): Pytorch model.
        X (torch.array): Network parameters.
    """
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):
    """Compute the accuracy of a model on a given dataset.

    Args:
        model (torch.nn): The Pytorch model.
        dataloader (torch.utils.data.DataLoader): Data loader of the dataset used for compting accuracy
        get_confusion_matrix (bool, optional): Generate confustion matrix. Defaults to False.
        device (str, optional): CPU or GPU. Defaults to "cpu".

    Returns:
        int: Accuracy of the model
    """

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
    """Save model to file.

    Args:
        model (torch.nn): Model to be saved.
        model_index (int): Client id that owns the current model.
        args (dict): Dictionary of user defined arguments.
    """
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    """Load model from file.

    Args:
        model (torch.nn): Model to be loaded.
        model_index (int): Client id that owns the current model.
        device (str, optional): CPU or GPU. Defaults to "cpu".

    Returns:
        torch.nn: Loaded model.
    """
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
    """Load validation data for the proxy similarity computation.

    Args:
        dataset (string): Name of the global dataset.
        datadir (string): Location of the global dataset.
        datasize (int): Number of samples to be drawn from the dataset.
        val_bs (int): Batch size for the data loader.

    Returns:
        torch.utils.data.DataLoader: Data loader for the validation data.
    """
    val_dl = None
    if dataset == 'tinyimagenet':
        if not os.path.exists('./data/tiny-imagenet-200'):
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip','./data/')
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
    """Generate train and test data loaders.

    Args:
        dataset (string): Dataset to be loaded.
        datadir (string): path to dataset.
        train_bs (int): Train data batch size.
        test_bs (int): Test data batch size.
        dataidxs (dict, optional): client:data-indeces. Defaults to None.
        noise_level (int, optional): Gaussian noise level. Defaults to 0.
        net_id (_type_, optional): Client id. Defaults to None.
        total (int, optional): _description_. Defaults to 0.

    Returns:
        tuple: Train and test dataloaders.
    """
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
    """Model wieght initializer.

    Args:
        m (torch.nn): Model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """Calculate the negative log likelihood of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """Sample random noise vector for training.

    Args:
        choice (_type_): _description_
        n_dis_c (_type_): Number of discrete latent code.
        dis_c_dim (_type_): Dimension of discrete latent code.
        n_con_c (_type_): Number of continuous latent code.
        n_z (_type_): Dimension of iicompressible noise.
        batch_size (_type_): Batch size.
        device (string): CPU or GPU.

    Returns:
        torch.ndarray: _description_
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
    """Flatten aand concatenate gradients from all layers of a model.

    Args:
        gradient (dict): Dictionary containing gradients.

    Returns:
        np.array: Flattened gradients.
    """
    X_flat = np.array([])
    for name,layer in sorted(gradient.items()):
        cur = np.array(layer,dtype=np.float64).flatten()
        X_flat = np.concatenate((X_flat,cur),axis=0)
    return X_flat

#Calculate the radians between two gradients
def get_radians(g1,g2):
    """Compute radian between two gradients.

    Args:
        g1 (np.array): Gradient A.
        g2 (np.array): Gradient B.

    Returns:
        float: Radian between two gradients.
    """
    unit_vector_1 = g1 / np.linalg.norm(g1) if np.linalg.norm(g1) != 0 else 0
    unit_vector_2 = g2 / np.linalg.norm(g2) if np.linalg.norm(g2) != 0 else 0
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    radians = np.arccos(dot_product)
    return radians

#BFTM methods
def show(matrix):
    """Display the similarity matrix.

    Args:
        matrix (list): 2-D similarity matrix.
    """
    print("",end="  ")
    for k in sorted(matrix.keys()):
        print(k,end="  ")
    
    for i,row in sorted(matrix.items()):
        print("\n" + str(i),end="  ")
        for j in row:
            print(matrix[i][j],end="  ")
    print()
        

def size(matrix):
    """Return the number of non None values in the matrix.

    Args:
        matrix (list): 2-D similarity matrix.

    Returns:
        int: Count of non None values in matrix.
    """
    size = 0
    for _,row in matrix.items():
        #size += len([r for r in row.values() if r != None])
        for _,v in row.items():
            #print("V:",v)
            size += 1 if v != None else 0
    return size

def is_complete(matrix,size):
    """Check whether the matrix is completely populated.

    Args:
        matrix (list): 2-D similarity matrix.
        size (int): Length of the 2-D similarity matrix.

    Returns:
        Boolean: True if the matrix is complete.
    """
    for _,row in matrix.items():
        if len(row) < size:
            return False
    return True

def exists(matrix,i,j):
    """Check if matrix value j exists in row i.
    This is used to determine if node j has similarity computed with node i.

    Args:
        matrix (list): 2-D similarity matrix.
        i (int): Matrix row.
        j (int): Value to locate.

    Returns:
        int: True if j and i have similarity computed.
    """
    return j in matrix[i].keys()

def intersection(matrix,n1,n2):
    """Collect overlap of encounter between two nodes.
    Encounters are nodes whoose similarity value has been computed.

    Args:
        matrix (list): 2-D node to node matrix.
        n1 (Node): Node one.
        n2 (Node): Node two.

    Returns:
        list: Return intersection.
    """
    inter = []
    for n,v in matrix[n1.id].items():
        if v == None and matrix[n2.id][n] == None:
            inter.append(n)
    return inter

def add_entry(matrix,i,j,replace=False):
    """Add entry to matrix.

    Args:
        matrix (list): 2-D similarity matrix.
        i (Node): Node one.
        j (Node): Node two.
        replace (bool, optional): Replace previous value of true. Defaults to False.
    """
    if j not in matrix[i].keys():
        matrix[i][j] = abs(i - j)
    else:
        if replace:
            matrix[i][j] = abs(i - j)

def get_preds(model, dataloader, device="cpu"):
    """Compute prediction on dataset.

    Args:
        model (torch.nn.Module): Model.
        dataloader (DataLoader): Dataset data loader.
        device (str, optional): CPU or GPU. Defaults to "cpu".

    Returns:
        list: Predictions on dataset.
    """

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
    """Compute the KL-divergence between a pair of predictions by
    a pair of models.

    Args:
        id1 (int): Node one id.
        id2 (int): Node two id.
        m1 (torch.nn.Module): Model one.
        m2 (torch.nn.Module): Model two.
        cache_val_preds (dict): Cached predictions
        val_dl (DataLoader): Validation dataset data loader.
        device (str, optional): CPU or GPU. Defaults to "cpu".

    Returns:
        float: KL-divergence between two predictions.
    """
    if m1 == m2:
        return 0
    
    m1_preds = F.log_softmax(get_preds(m1,val_dl,device),dim=1) if cache_val_preds[id1] == None else cache_val_preds[id1]
    m2_preds = F.log_softmax(get_preds(m2,val_dl,device),dim=1) if cache_val_preds[id2] == None else cache_val_preds[id2]

    cache_val_preds[id1] = m1_preds
    cache_val_preds[id2] = m2_preds

    print("PRED SHAPE:", m1_preds.shape)

    kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)

    result = kl_loss(m1_preds, m2_preds)
    print("\t",result,"dtype:",result.dtype)

    return result

def get_signed_radians(grad1,grad2):
    """Compute signed radians between two gradients.

    Args:
        grad1 (np.ndarray): Gradient one.
        grad2 (np.ndarray): Gradient two.

    Returns:
        float: Signed angle.
    """
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
    """Update similarity matrix.

    Args:
        G (nx.graph): DL graph.
        M (list): 2-D similarity matrix.
        C (dict): Node encounter cache contains the ids of previously encountered nodes.
        cache_val_preds (dict): Cached prediction outputs on the val dataset.
        adj_list (list): DL adjacency list.
        sim (str, optional): How to compare the nodes, kl-divergence or gradient similarity. Defaults to "grad".
        val_dl (_type_, optional): Validation dataset data loader. Defaults to None.
        device (str, optional): CPU or GPU. Defaults to "cpu".
    """
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
    """Breadth-First-Topology-Morphing method.

    Args:
        G (nx.graph): DL graph.
        M (list): Similarity matrix.
        C (dict): Node encounter cache.
        degree (int): Degree of the graph. (Max number of edges for a node)

    Returns:
        nx.graph: DL training graph.
    """
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
    """Check if empty cluster exists.

    Args:
        clusters (list): List of clusters.

    Returns:
        bool: True if empty cluster exists.
    """
    for _,cluster in clusters.items():
        if len(cluster) > 0:
            return False
    return True

def greedy_cliques(cliques,clusters,matrix):
    """Construct a final topology using greedy search.
    The greedy search is used to select nodes that form
    a clique.

    Args:
        cliques (list): List of cliques.
        clusters (list): List of clusters.
        matrix (list): 2-D similarity matrix.

    Returns:
        list: Cliques.
    """
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
    """Construct as many cliques as there are clusters.

    Args:
        clusters (list): List of clusters each containing nodes.
        strategy (string): Strategy on how to construct the topology.
        labels (list): The cluster information for each node.
        cut (int, optional): Number of partitions in the topology. Defaults to 0 means just one partition.
        matrix (list, optional): Similarity matrix. Defaults to None.

    Returns:
        nx.graph: DL topology.
    """
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
    """Create a single clique by selecting nodes from
    only one cluster when strategy=rand, or selecting
    one node from each cluster when strategy=optim.

    Args:
        clusters (list): List of clusters containing nodes.
        strategy (string): How to form the clique.

    Returns:
        nx.graph: DL topology.
    """
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
    """Split cliques into 'cut' partitions.

    Args:
        cliques (list): List of cliques.
        cut (int, optional): Number of partitions. Defaults to 1.

    Returns:
        list: List of partitions.
    """
    cliques.sort(key=len)
    k, m = divmod(len(cliques), cut)
    return list(cliques[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(cut))

def cliques_on_ring(cliques,labels,G, cut=0):
    """Connect cliques in a ring structure to create
    a ring of cliques topology.

    Args:
        cliques (list): List of cliques.
        labels (list): Node labels indicating to which cluster they belong.
        G (nx.partition): DL graph.
        cut (int, optional): Number of partitions. Defaults to 0 meaning only 1 partition.
    """
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
    """Create DL graph given adjacency list and cliques.

    Args:
        adj_list (list): Adjacency list of nodes.
        cliques (list): List of cliques.

    Returns:
        nx.graph: DL graph.
    """
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
    """Given nodes and the clusters to which they are assigned,
    construct a DL graph madeup of cliques.

    Args:
        adj_list (list): Adjacency list of DL nodes.
        labels (list): Labeled clusters.
        matrix (list): Similarity matrix.
        topology (str, optional): Type of topology to create. Defaults to "clique".
        cut (int, optional): Number of partitions in the DL graph. Defaults to 0 which means no cuts i.e. single partition.

    Returns:
        nx.graph: DL topology.
    """
    num_clusters = list(np.unique(labels))
    clusters = {i:[] for i in num_clusters}
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
    """Experimental BFTM for tree topology.

    Args:
        adj_list (dict): DL graph adjacency list.
        labels (np.array): The cluster labels for each client.

    Returns:
        nx.graph: DL network graph.
    """
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


def init_nets(net_configs, dropout_p, n_parties, args):
    """Initialize network

    Args:
        net_configs (_type_): _description_
        dropout_p (float): Dropout ratio.
        n_parties (int): Number of clients.
        args (dict): User options.

    Returns:
        tuple: models, model_meta_data, layer_type
    """

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-ffnn":
            #Assuming we are using MNIST dataset: input size = 784, num_classes = 10
            input_size, hidden_size, num_classes = 784, 10, 10
            net = NeuralNet(input_size, hidden_size, num_classes)
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            net = ResNet50_cifar10()
        elif args.model == "res18":
            net = torchvision.models.resnet18(pretrained=args.pretrained == 1)
            #Finetune Final layers to adjust for tiny imagenet input
            if args.dataset == "tinyimagenet":                
                net.avgpool = nn.AdaptiveAvgPool2d(1)
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, 200)
            elif args.dataset == "cifar100":
                net.avgpool = nn.AdaptiveAvgPool2d(1)
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, 100)
            elif args.dataset in ["cifar10"]:
                net.avgpool = nn.AdaptiveAvgPool2d(1)
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, 10)
        elif args.model == 'res20':
            net = resnet20()
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, sched, device="cpu",stash=False,accuracy=True):
    """Train single model.

    Args:
        net_id (int): Client id.
        net (torch.Module): Client model.
        train_dataloader (DataLoader): Train dataset data loader.
        test_dataloader (DataLoader): Test dataset data loader.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        args_optimizer (dict): User options.
        sched (Bool): Use schedular if set to True.
        device (str, optional): CPU or GPU. Defaults to "cpu".
        stash (bool, optional): Stash last layer gradients if set to True. Defaults to False.
        accuracy (bool, optional): Compute local accuracy if set to True. Defaults to True.

    Returns:
        tuple: Train and Test accuracy.
    """
    logger.info('Training network %s' % str(net_id))
    train_acc, test_acc = None, None
    if accuracy:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)#, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        last_td = len(train_dataloader) - 1
        # print ("Epoch", epoch)
        for idx,tmp in enumerate(train_dataloader):
            last_batch = len(tmp) - 1
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                # print ("\t batch " + str(batch_idx) +" trained")
                #Collect grads here if in pretraining (stash) mode!
                if stash and epoch == epochs - 1 and idx == last_td and batch_idx == last_batch:
                    net.stash_grads()
                
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
        
        if sched:
            scheduler.step()

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    if accuracy:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def view_image(train_dataloader):
    """View single image in train dataset.

    Args:
        train_dataloader (DataLoader): Train data loader.
    """
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)

def local_pre_training(nets, selected, args, net_dataidx_map, pre_epochs, test_dl = None, device="cpu"):
    """Perform prologue training.

    Args:
        nets (dict): Total list of models to train.
        selected (list): List of selected models.
        args (dict): User options.
        net_dataidx_map (dict): Client:data-indices map.
        pre_epochs (int): Number of training epochs.
        test_dl (DataLoader, optional): Test data loader. Defaults to None.
        device (str, optional): CPU or GPU. Defaults to "cpu".
    """
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        sched = args.model in ["res20", "vgg"]

        #Pre train for 10 epochs
        _, _ = train_net(net_id, net, train_dl_local, test_dl, pre_epochs, args.lr, args.optimizer, sched, device=device, stash=args.similarity == "grad",accuracy=args.local_acc)

def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    """Peform model updates for the clients in the DL network.

    Args:
        nets (dict): Clients networks.
        selected (list): List of currently selected models.
        args (dict): User options.
        net_dataidx_map (dict): Client:data-indices map.
        test_dl (DataLoader, optional): Test data loader. Defaults to None.
        device (str, optional): CPU or GPU. Defaults to "cpu".

    Returns:
        tuple: models, average train accuracy, average test accuracy
    """
    avg_acc, avg_train_acc = None, None
    if args.local_acc:
        avg_acc = 0.0
        avg_train_acc = 0.0
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        sched = args.model in ["res20", "vgg","res18"]

        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, sched, device=device, accuracy=args.local_acc)
        if args.local_acc:
            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc
            avg_train_acc += trainacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    
    if args.local_acc:
        avg_acc /= len(selected)
        avg_train_acc /= len(selected)
        if args.alg == 'local_training':
            logger.info("avg train acc %f \t avg test acc %f" % (avg_train_acc,avg_acc))

    nets_list = list(nets.values())
    return nets_list, avg_train_acc, avg_acc

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    """This method returns a dictionary with node_id keys and a list of data_id indexes.

    Args:
        dataset (string): Dataset name.
        partition (string): Parrtition type.
        n_parties (int): Number of clients.
        init_seed (int, optional): Random seed. Defaults to 0.
        datadir (str, optional): Dataset path. Defaults to './data'.
        logdir (str, optional): Output directory path. Defaults to './logs'.
        beta (float, optional): Non-IID concentration parameter. Defaults to 0.5.

    Returns:
        dict: Client:data-indices map.
    """
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

def average_gradients(grads):
    """Average model gradients.

    Args:
        grads (list): List of model gradients.

    Returns:
        list: Averaged gradients.
    """
    avg_grad = copy.deepcopy(grads[0])
    for idx,grad in enumerate(grads):
        if idx >= 1:
            for name,layer in grad.items():
                avg_grad[name] += layer
    for name,layer in avg_grad.items():
        avg_grad[name] /= len(grads)
    return avg_grad

#This method averages weights in w list
def average_weights(w):
    """Returns the average of the weights.

    Args:
        w (torch.nn.Module): Model weights.

    Returns:
        torch.nn.Module: Averaged weight.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def ripple_updates(G):
    """Ripple updates with 1-hop neighbors. 
    Compute vicinity weights for all nodes in graph G.

    Args:
        G (nx.graph): Peer graph
    """
    weight_updates = {n:None for n in list(G.nodes)}
    for node in list(G.nodes):
        vicinity_weights = [copy.deepcopy(node.model.state_dict())]
        for neighbor,_ in G.adj[node].items():
            vicinity_weights.append(copy.deepcopy(neighbor.model.state_dict()))
        vicinity_weights = average_weights(vicinity_weights)
        weight_updates[node] = vicinity_weights
    
    #Update vicinity weights
    for node,weight in weight_updates.items():
        node.model.load_state_dict(weight)