import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os

from math import *
import random
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from node import *
from resnet import *
from dlrl.dlrl import gen_cliques


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--pre_epochs', type=int, default=10, help='number of local pre train epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--pretrained', type=int, default=0, help='Load pretrained model')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--local_acc', type=int, default=1, help='Enable local accuracy collection [0,1]')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--topology',type=str, default="pcc", help='Node graph topology default=pcc, options (ring, clique, tree)')
    parser.add_argument('--strategy',type=str, default="rand", help='Clumping strategy default=rand, options (optim,greedy,RL)')
    parser.add_argument('--similarity',type=str, default="kl", help='Similarity computation default=kl, options (grad))') #TODO: emd similarity
    parser.add_argument('--cut',type=int, default=0, help='Number of cuts in final graph')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    #Extract user arguments
    NODES = args.n_parties
    MAX_PEERS = NODES - 1 if args.topology == "complete" else math.ceil(math.log(NODES,2))
    SIM_MATRIX = {i:{j: 0 if i == j else None for j in range(NODES)} for i in range(NODES)}
    TOPOLOGY = args.topology
    STRATEGY = args.strategy
    NOW = str(datetime.datetime.now()).replace(" ","--")
    IID = args.partition
    SIMILARITY = args.similarity
    CUT = args.cut
    DATASET = args.dataset
    NOISE = args.noise

    #Create random template graph with max degree size
    GT = nx.random_regular_graph(d=MAX_PEERS, n=NODES)


    #Create output directory
    args.logdir = './logs/{}_{}_{}_{}_nodes[{}]_maxpeers[{}]_rounds[{}]_topology[{}]_strategy[{}]_cut[{}]_similarity[{}]_frac[{}]_local_ep[{}]_local_bs[{}]_beta[{}]_noise[{}]/'. \
        format(NOW,args.dataset, args.model, IID, NODES, MAX_PEERS, args.comm_round,TOPOLOGY,STRATEGY,CUT,SIMILARITY, args.sample,args.epochs, args.batch_size, args.beta, args.noise)
    os.makedirs(os.path.dirname(args.logdir), exist_ok=True)

    
    #Create log files
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Device: %s' % str(device))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    #Setting random seeds for experimental reproducibility
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    #Partition data based on user input
    logger.info("Partitioning data")
    print("Datadir:", args.datadir)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    #Get the number of labels
    n_classes = len(np.unique(y_train))
    args.datadir = args.datadir + "tiny-imagenet-200/" if args.dataset == "tinyimagenet" else args.datadir
    
    # print("#classes:",n_classes)
    # print("#X_train:",len(X_train))
    # print("#X_test:",len(X_test))
    # print("#netidxmap",len(net_dataidx_map))
    # for i,net in enumerate(net_dataidx_map):
    #     print("\t#netidxmap["+str(i)+"]",len(net_dataidx_map[i]))

    #Setup train and test dataloaders
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    # print("len train_ds_global:", len(train_ds_global) if train_ds_global else train_ds_global)
    # print("len test_ds_global:", len(test_ds_global) if test_ds_global else test_ds_global)

    #Setup global dataset dataloader
    if DATASET in ["cifar10","cifar100"]:
        val_dl_global = get_val_dataloader("tinyimagenet", "./data/tiny-imagenet-200/", 1000, 32)
    elif DATASET in ["mnist","femnist"]:
        val_dl_global = get_val_dataloader("fmnist", "./data/", 1000, 32)
    elif DATASET in ["tinyimagenet"]:
         val_dl_global = get_val_dataloader("cifar10", "./data/", 1000, 32)

    # print("len train_dl_global:",len(train_dl_global))
    # print("len val_dl_global:",len(val_dl_global))
    # print("len test_dl_global:",len(test_dl_global))

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)


    #Add Gaussian noise to data if required
    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)


    #Initialize the client models
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
    # global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
    # global_model = global_models[0]
    # print(nets[0])

    #Initialize peer adjacency list
    adj_list = [Node(idx,net,net_dataidx_map[idx],MAX_PEERS) for idx,net in nets.items()]

    #Initialize node caches
    CACHE = {n.id:set() for n in adj_list}

    #Initialize val preds cache
    cache_val_preds = {n.id:None for n in adj_list}

    #Construct G0 using the template graph
    G0 = nx.Graph()
    for u,v in GT.edges:
        G0.add_edge(adj_list[u],adj_list[v])
    
    # print("TOPOLOGY:",TOPOLOGY)
    
    if TOPOLOGY not in ["complete"]:
        #Pretrain selected nodes to compute local gradients for 10 epochs
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.sample)]
        local_pre_training(nets, selected, args, net_dataidx_map, args.pre_epochs, test_dl = test_dl_global, device=device)

        #Topology morphing
        update_matrix(G0,SIM_MATRIX,CACHE,cache_val_preds,adj_list,sim=SIMILARITY,val_dl=val_dl_global,device=device)

        count = 1
        while size(SIM_MATRIX) < NODES ** 2:
            print("\nMATRIX i="+ str(count) +": size=",size(SIM_MATRIX))
            G0 = BFTM(G0,SIM_MATRIX,CACHE,MAX_PEERS)
            update_matrix(G0,SIM_MATRIX,CACHE,cache_val_preds,adj_list,sim=SIMILARITY,val_dl=val_dl_global,device=device)
            count += 1

        #show(SIM_MATRIX)
        print("\nFINAL MATRIX i="+ str(count - 1) +": size=",size(SIM_MATRIX))

        matrix = []
        for _,row1 in sorted(SIM_MATRIX.items()):
            temp = []
            for val,row2 in sorted(row1.items()):
                temp.append(float(row2))
            matrix.append(temp)
        
        logger.info("SIM MATRIX: %s", matrix)

        #print(matrix)
        matrix = np.array(matrix)
        print("matrix.shape:",matrix.shape)

        print("Clustering sim matrix...")
        #Cluster peers
        kmeans = KMeans(n_clusters=MAX_PEERS, random_state=0).fit(matrix)

        labels = {i:[] for i in range(MAX_PEERS)}
        for idx,label in enumerate(kmeans.labels_):
            labels[label].append(adj_list[idx])

        print("Labels:")
        for k,v in labels.items():
            print("\t",k,":",len(v))
        print("\tTotal:",len(kmeans.labels_))

        #Plot clusters
        pca = PCA(2)
        print("matrix.shape",matrix.shape)
        x_rads = matrix
        print("x_rads.shape:",x_rads.shape)
        df_rads = pca.fit_transform(x_rads)
        print("df_rads.shape",df_rads.shape)
        label_rads = kmeans.labels_
        print("label_rads.shape",label_rads.shape)
        u_labels_rads = np.unique(label_rads)
        for i in u_labels_rads:
            plt.scatter(df_rads[label_rads == i , 0] , df_rads[label_rads == i , 1] , label = i)
        plt.legend()
        plt.savefig(args.logdir + "kmeans.png")
        plt.close()
        #plt.show()

        #Generate random labels for random strategy
        #This simply shuffles the clustered nodes
        rand_labels = [i for i in list(kmeans.labels_)]
        random.shuffle(rand_labels)

        if TOPOLOGY == "tree":
            if STRATEGY == "rand":
                G0 = BFTM_(adj_list,rand_labels)
            else:
                G0 = BFTM_(adj_list,list(kmeans.labels_))
        elif TOPOLOGY == "clique":
            if STRATEGY == "rand":
                G0 = m_cliques(adj_list,rand_labels,SIM_MATRIX)
            else:
                G0 = m_cliques(adj_list,list(kmeans.labels_),SIM_MATRIX)
        # pcc (per cluster clique) topology creates one clique per cluster
        # The cliques will be arranged in a ring topology
        # This topology is the most suitable for comparisons
        elif TOPOLOGY == "pcc":
            #The rand option creates a ring of locally homogenous cliques
            #Here, local cliques are created via uniform sampling from among kmeans clusters 
            if STRATEGY == "rand":
                G0 = m_cliques(adj_list,list(kmeans.labels_),SIM_MATRIX,"pcc_rand",CUT)
            #The greedy option creates a ring of locally heterogeneous cliques
            #Here, local cliques are created via greedy construction from among kmeans clusters
            elif STRATEGY == "greedy":
                G0 = m_cliques(adj_list,list(kmeans.labels_),SIM_MATRIX,"pcc_greedy",CUT)
            elif STRATEGY == "RL": 
                topo = gen_cliques(matrix,5)
                G0 = manual_cliques(adj_list,topo)
            #The optim option creates locally heterogenous cliques
            #Here, local cliques are created via uniform sampling from among kmeans clusters
            else:
                G0 = m_cliques(adj_list,list(kmeans.labels_),SIM_MATRIX,"pcc_optim",CUT)
            #We have added a manual topology option to go with a manual partition
        #At the moment, this option is not yet programmable as it only considers 24 nodes
        elif TOPOLOGY == "manual":
            print("Manual topology!")
            # The optim strategy creates locally heterogenous set of cliques
            if STRATEGY == "optim":
                topo = [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23]]
                print("Optim topology!")
                G0 = manual_cliques(adj_list,topo)
            #The rand strategy creates cliques madeup of random nodes
            elif STRATEGY == "rand":
                topo = [i for i in range(24)]
                random.shuffle(topo)
                topo = np.array_split(topo,5)
                G0 = manual_cliques(adj_list,topo)
                print("Rand topology!")
            #RL strategy creates cliques using the RL agent trained on graphs of degree 5
            elif STRATEGY == "RL": 
                topo = gen_cliques(matrix,5)
                G0 = manual_cliques(adj_list,topo)
                print("RL topology!")
            # nx.draw(G0, with_labels = True)
            # plt.savefig(args.logdir + "graph.png")
        elif TOPOLOGY == "sample":
            if STRATEGY == "rand":
                G0 = m_cliques(adj_list,list(kmeans.labels_),SIM_MATRIX,"sample_rand")
            else:
                G0 = m_cliques(adj_list,list(kmeans.labels_),SIM_MATRIX,"sample_optim")
            print("Sampling done. G0.nodes:",G0.number_of_nodes())
            print("G0 labels: ",[kmeans.labels_[n.id] for n in list(G0.nodes)])
        else:
            if STRATEGY == "rand":
                G0 = m_cliques(adj_list,rand_labels,"ring",SIM_MATRIX)
            else:
                G0 = m_cliques(adj_list,list(kmeans.labels_),"ring",SIM_MATRIX)

        nx.draw(G0,node_color=[kmeans.labels_[n.id]/len(kmeans.labels_) for n in list(G0.nodes)],with_labels = True)
        plt.savefig(args.logdir + "graph.png")
    else:
        nx.draw(G0)
        plt.savefig(args.logdir + "graph.png")
    
    #Edit below to initialize all peers with same random weights
    # global_para = global_model.state_dict()
    # if args.is_same_initial:
    #     for net_id, net in nets.items():
    #         net.load_state_dict(global_para)


    #Training commences once topological pre-processing is over
    #This is main DL training loop where nodes are trained in each communication round
    for round in range(args.comm_round):
        logger.info("in comm round:" + str(round))

        #Select all nodes for local training
        subnets = {net_i:net for net_i,net in nets.items() if net in [n.model for n in list(G0.nodes)]}
        selected = [net_i for net_i,_ in subnets.items()]
        _, avg_local_train_acc, avg_local_test_acc = local_train_net(subnets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)

        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl_global))

        #Perform local aggregation by averaging neighborhood weights
        ripple_updates(G0)

        avg_global_train_acc, avg_global_test_acc = 0.0, 0.0
        for idx,net in subnets.items():
            avg_global_train_acc += compute_accuracy(net, train_dl_global,get_confusion_matrix=False, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dl_global, get_confusion_matrix=True, device=device)
            avg_global_test_acc += test_acc
        
        avg_global_train_acc /= len(subnets)
        avg_global_test_acc /= len(subnets)

        if args.local_acc:
            logger.info('>> Avg Local Train accuracy: %f' % avg_local_train_acc)
            logger.info('>> Avg Local Test accuracy: %f' % avg_local_test_acc)
        logger.info('>> Avg Global Train accuracy: %f' % avg_global_train_acc)
        logger.info('>> Avg Global Test accuracy: %f' % avg_global_test_acc)
