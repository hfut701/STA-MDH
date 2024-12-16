import argparse


def arg_parameter():
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument('--device', type=str, default='cuda:1', help='')
    parser.add_argument('--model', type=str, default='none', help='---')
    parser.add_argument('--dataset', type=str, default='park', help="crosscheck or park or ppmi")
    parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits')
    parser.add_argument('--seed', type=int, default=5400, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate coress:0.01')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    parser.add_argument('--reg', type=int, default=1, help='enable regulizer or not for local train')
    parser.add_argument('--com_round', type=int, default=100, help='Number of communication round to train.')
    parser.add_argument('--epoch', type=int, default=20, help='epoch for each communication round.')
    parser.add_argument('--client_epochs', type=int, default=20, help='epoch for each communication round.')
    parser.add_argument('--logDir', default='./log/,default.txt', help='Path for log info')
    parser.add_argument('--num_thread', type=int, default=5, help='number of threading to use for client training.')
    parser.add_argument('--dataaug', type=int, default=0, help='data augmentation')
    parser.add_argument('--evalall', type=int, default=0, help='use all or partial validation dataset for test')

    # Federated arguments
    parser.add_argument('--clients', type=int, default=51, help="number of users: K:10,20,31,41,51")
    parser.add_argument('--shards', type=int, default=2, help="each client roughly have 2 data classes")
    parser.add_argument('--serveralpha', type=float, default=1, help='server prop alpha')
    parser.add_argument('--serverbeta', type=float, default=0.3, help='personalized agg rate alpha')
    parser.add_argument('--deep', type=int, default=0, help='0: 1 layer only, 1: 2 layers, 3:full-layers')
    parser.add_argument('--agg', type=str, default='fen_attation_gat', help='-----')
    parser.add_argument('--dp', type=float, default=0.005, help='differential privacy')
    parser.add_argument('--epsilon', type=float, default=1, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    parser.add_argument('--sm', type=str, default='full', help='state mode, for baselines running')
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--client_frac', type=float, default=1, help='the fraction of clients')

    # Graph Learning
    parser.add_argument('--subgraph_size', type=int, default=30, help='k')
    parser.add_argument('--adjalpha', type=float, default=3, help='adj alpha')
    parser.add_argument('--gc_epoch', type=int, default=10, help='')
    parser.add_argument('--adjbeta', type=float, default=0.5, help='update ratio')
    parser.add_argument('--edge_frac', type=float, default=1, help='the fraction of clients')




    args = parser.parse_args()

    hidden = []
    for h in args.hidden.split(","):
        hidden.append(int(h))
    args.hidden = hidden
    logs = args.logDir.split(",")
    args.logDir = logs[0] + args.dataset + "-" + str(args.client_frac) + "-" + \
                  str(args.com_round) + "-" + str(args.client_epochs) + "-" +'true'+ logs[1]
    return args
