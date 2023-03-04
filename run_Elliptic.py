import time
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.pytorchtools import EarlyStopping
from utils.data import load_Elliptic_data
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch,feature_df_plot
from model import MAGNN_nc_mb
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore", category=Warning)
out_dim = 2
dropout_rate = 0.3
lr = 0.05
weight_decay = 0.001
#TAT,TBT,TCT
#TA=0,AT=1,TB=2,BT=3,TC=4,CT=5
etypes_num = 6
etypes_list = [[0,1], [2,3], [4, 5]]
print("out_dim:{0},dropout_rate:{1},lr:{2}".format(out_dim,dropout_rate,lr))
def train_model_Elliptic(time_step,feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_Elliptic_data(time_step)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = np.concatenate([train_val_test_idx['train_idx'],train_val_test_idx['test_idx'],train_val_test_idx['test_idx']],axis=0)
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []
    for _ in range(repeat):
        # num_metapaths:3 TCT,TBT,TAT
        # num_edge_type:6 TC CT TB BT TA AT
        net = MAGNN_nc_mb(3, etypes_num, etypes_list, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            test_embeddings = []
            net.train()
            for iteration in range(train_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)

                t1 = time.time()
                dur1.append(t1 - t0)

                logits, embeddings = net((train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch])
                test_embeddings.append(embeddings)

                t2 = time.time()
                dur2.append(t2 - t1)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)
                if iteration % 50 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

            # validation
            net.eval()
            val_logp = []

            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                    logits, embeddings = net(
                        (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    logp = F.log_softmax(logits, 1)
                    val_logp.append(logp)

                val_loss = F.nll_loss(torch.cat(val_logp, 0), labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                test_embeddings = torch.cat(test_embeddings, 0)
                if(time_step==34):
                    np.save("./emb/train_MAGNN_emb.npy", test_embeddings.detach().numpy())
                    np.save("./emb/train_labels.npy", labels[train_idx].detach().numpy())
                else:
                    np.save("./emb/test_MAGNN_emb.npy", test_embeddings.detach().numpy())
                    np.save("./emb/test_labels.npy", labels[train_idx].detach().numpy())
                break

def test_data(alldata,time_step,feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    lf = tx_features
    af = (tx_features + agg_features)
    if(time_step==50):
        feadata = alldata[(alldata['time_step'] <= 49) &(alldata['time_step'] >= 35) & (alldata['class'] != 2)]
    else:
        feadata = alldata[(alldata['time_step'] == time_step) & (alldata['class'] != 2)]
    features0_af = np.array(feadata[af])
    features0_lf = np.array(feadata[lf])
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_Elliptic_data(time_step)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    test_idx = np.concatenate([train_val_test_idx['train_idx'],train_val_test_idx['val_idx'],train_val_test_idx['test_idx']],axis=0)
    test_idx = np.sort(test_idx)
    # testing with evaluate_results_nc
    test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
    net = MAGNN_nc_mb(3, etypes_num, etypes_list, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
    net.to(device)
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
    net.eval()
    test_embeddings = []
    with torch.no_grad():
        for iteration in range(test_idx_generator.num_iterations()):
            # forward
            test_idx_batch = test_idx_generator.next()
            test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(adjlists,
                                                                                         edge_metapath_indices_list,
                                                                                         test_idx_batch,
                                                                                         device, neighbor_samples)
            logits, embeddings = net(
                (test_g_list, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
            test_embeddings.append(embeddings)
        test_embeddings = torch.cat(test_embeddings, 0)
        np.save("./emb/test_MAGNN_emb.npy",test_embeddings.cpu().numpy())
        np.save("./emb/test_labels.npy",labels[test_idx].cpu().numpy())
        # svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(test_embeddings.cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
        #test_features = test_embeddings.cpu().numpy()
        print("MAGNN+AF")
        test_features_afne = np.concatenate([test_embeddings.cpu().numpy(), features0_af],axis=1)
        evaluate_results_nc(test_features_afne, labels[test_idx].cpu().numpy(), num_classes=out_dim,modelname="MAGNN+AF")
        print("MAGNN+LF")
        test_features_lfne = np.concatenate([test_embeddings.cpu().numpy(), features0_lf], axis=1)
        evaluate_results_nc(test_features_lfne, labels[test_idx].cpu().numpy(), num_classes=out_dim,modelname="MAGNN+LF")

        print("AF")
        evaluate_results_nc(features0_af, labels[test_idx].cpu().numpy(), num_classes=out_dim,modelname="AF")
        #print("LF")
        #evaluate_results_nc(features0_lf, labels[test_idx].cpu().numpy(), num_classes=out_dim,modelname="LF")
        print("MAGNN")
        evaluate_results_nc(test_embeddings.cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim,modelname="MAGNN")
def test_data_sin(alldata,time_step,feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    lf = tx_features
    af = (tx_features + agg_features)
    if(time_step==50):
        feadata = alldata[(alldata['time_step'] <= 49) &(alldata['time_step'] >= 35) & (alldata['class'] != 2)]
    else:
        feadata = alldata[(alldata['time_step'] == time_step) & (alldata['class'] != 2)]
    features0_af = np.array(feadata[af])
    features0_lf = np.array(feadata[lf])
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_Elliptic_data(time_step)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    in_dims = [features.shape[1] for features in features_list]
  
    labels = torch.LongTensor(labels).to(device)
    test_idx = np.concatenate([train_val_test_idx['train_idx'],train_val_test_idx['val_idx'],train_val_test_idx['test_idx']],axis=0)
    test_idx = np.sort(test_idx)
    # testing with evaluate_results_nc
    test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
    net = MAGNN_nc_mb(3, etypes_num, etypes_list, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
    net.to(device)
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
    net.eval()
    test_logp = []
    test_embeddings = []
    with torch.no_grad():
        for iteration in range(test_idx_generator.num_iterations()):
            # forward
            test_idx_batch = test_idx_generator.next()
            test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(adjlists,
                                                                                         edge_metapath_indices_list,
                                                                                         test_idx_batch,
                                                                                         device, neighbor_samples)
            logits, embeddings = net(
                (test_g_list, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
            test_embeddings.append(embeddings)
            logp = F.log_softmax(logits, 1)
            test_logp.append(logp)
        val_loss = F.nll_loss(torch.cat(test_logp, 0), labels[test_idx])
        print(val_loss)
        test_embeddings = torch.cat(test_embeddings, 0)

def evaluate_embedding():
    feadata_test = alldata[(alldata['time_step'] <= 49) &(alldata['time_step'] >= 35) & (alldata['class'] != 2)]
    feadata_train = alldata[(alldata['time_step'] < 35) & (alldata['class'] != 2)]
    af = (tx_features + agg_features)
    features_af_train = np.array(feadata_train[af])
    features_af_test = np.array(feadata_test[af])
    train_embeddings = np.load("./emb/train_MAGNN_emb.npy")
    test_embeddings = np.load("./emb/test_MAGNN_emb.npy")
    train_labels = np.load("./emb/train_labels.npy")
    test_labels = np.load("./emb/test_labels.npy")

    print("MAGNN+AF")
    modelname = "MAGNN+AF"
    print(train_embeddings.shape,features_af_train.shape)
    print(test_embeddings.shape, features_af_test.shape)
    X_train = np.concatenate([train_embeddings, features_af_train], axis=1)
    y_train = train_labels
    X_test = np.concatenate([test_embeddings, features_af_test], axis=1)
    y_test = test_labels
    rfc = RandomForestClassifier(random_state=0, n_estimators=100, max_features=100, n_jobs=-1, class_weight="balanced")
    rfc = rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    features = ["EMB_{}".format(i) for i in range(512)] + af
    feature_importances = rfc.feature_importances_
    features_df = pd.DataFrame({'Features': features, 'Importance': feature_importances})
    features_df.sort_values('Importance', inplace=True, ascending=False)
    plt = feature_df_plot(features_df, modelname)
    pos_probs = rfc.predict_proba(X_test)
    np.save("./predict_res/{0}-{1}.npy".format(modelname, test_size), np.array(pos_probs))
    np.save("./predict_res/ytrue-{0}.npy".format(test_size), np.array(y_test))
    Precision, recall, f1score, num = precision_recall_fscore_support(y_test, y_pred, average=None)

    print("AF")

    print("MAGNN")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the Elliptic dataset')
    ap.add_argument('--feats-type', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=64 , help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn_vec_dim', type=int, default=16, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn_type', default='TransE2', help='Type of the aggregator. Default is TransE2.')
    ap.add_argument('--epoch', type=int, default=200, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=512, help='Batch size. Default is 256.')
    ap.add_argument('--samples', type=int, default=50, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save_postfix', default='Elliptic', help='Postfix for the saved model and result. Default is DBLP.')

    args = ap.parse_args()
    print(args)
    #train_model_Elliptic(34,args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,args.epoch, args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)
    tx_features = ["tx_feat_" + str(i) for i in range(2, 95)]
    agg_features = ["agg_feat_" + str(i) for i in range(1, 73)]
    features = pd.read_csv("F:\AML\MyWork\Data_process\elliptic_txs_features_new.csv")
    classes = pd.read_csv("F:\AML\MyWork\Data_process\elliptic_txs_classes_new.csv")
    alldata = pd.merge(features, classes, on='txId')
    test_data(alldata,50, args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,args.epoch, args.patience, args.batch_size, args.samples, args.repeat, save_postfix='Elliptic')