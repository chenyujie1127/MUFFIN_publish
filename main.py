from operator import mul
import random
import numpy as np
import torch
from time import time
import copy
import logging
import argparse
import torch.optim as optim
import torch.utils.data as Data
from utils import *
from models import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_SKGDDI_args():
    parser = argparse.ArgumentParser(description="Run MUFFIN.")

    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='DRKG',
                        help='Choose a dataset from {DrugBank, DRKG}')
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--graph_embedding_file', nargs='?', default='data/DRKG/gin_supervised_masking_embedding.npy',
                        help='Input data path.')
    parser.add_argument('--entity_embedding_file', nargs='?', default='data/DRKG/DRKG_TransE_l2_entity.npy',
                        help='Input data path.')
    parser.add_argument('--relation_embedding_file', nargs='?', default='data/DRKG/DRKG_TransE_l2_relation.npy',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='1: Pretrain with the learned embeddings, 2:use pretrain_model_path')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--DDI_batch_size', type=int, default=2048,
                        help='DDI batch size.')
    parser.add_argument('--DDI_evaluate_size', type=int, default=2500,
                        help='KG batch size.')

    parser.add_argument('--entity_dim', type=int, default=100,
                        help='User / entity Embedding size.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--ddi_print_every', type=int, default=1,
                        help='Iter interval of printing DDI loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating DDI.')

    parser.add_argument('--multi_type', nargs='?', default='False',
                        help='whether task is multi-class')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='FC output dim: 81 or 1')

    parser.add_argument('--n_hidden_1', type=int, default=2048,
                        help='FC hidden 1 dim')
    parser.add_argument('--n_hidden_2', type=int, default=2048,
                        help='FC hidden 2 dim')
    
    parser.add_argument('--in_channels', type=int, default=1,
                        help='in_channels')
    parser.add_argument('--out_channels', type=int, default=8,
                        help='out_channels')
    parser.add_argument('--kernel', type=int, default=5,
                        help='kernel_size')
    parser.add_argument('--pooling_size', type=int, default=2,
                            help='pooling_size')

    parser.add_argument('--structure_dim', type=int, default=300,
                        help='structure_dim')
    parser.add_argument('--pre_entity_dim', type=int, default=400,
                        help='pre_entity_dim')

    parser.add_argument('--feature_fusion', nargs='?', default='double',
                        help='feature fusion type: concat / sum / double')

    args = parser.parse_args()

    save_dir = 'trained_model/MUFFIN/epoch_{}/{}/all_entitydim{}_feature{}_lr{}_muti_type{}/'.format(args.n_epoch,
        args.data_name, args.entity_dim, args.feature_fusion, args.lr, args.multi_type)
    args.save_dir = save_dir

    return args


def calc_metrics(y_true, y_pred, pred_score, multi_type):
    
    acc = accuracy(y_true, y_pred)
    precision_score = precision(y_true, y_pred,multi_type)
    recall_score = recall(y_true, y_pred,multi_type)
    f1_score = f1(y_true, y_pred,multi_type)
    auc_score = None if multi_type != 'False' else auc(y_true.cuda().data.cpu().numpy(),pred_score.cuda().data.cpu().numpy())

    return acc,precision_score,recall_score,f1_score,auc_score

def evaluate(args, model, loader_test, loader_idx,):

    model.eval()

    precision_list, recall_list, f1_list, acc_list, auc_list= [],[],[],[],[]

    with torch.no_grad():

        for data in loader_test:
            
            test_x, test_y = data
            out, all_embedding = model('predict', test_x, loader_idx)
            
            if args.multi_type == 'False':
                out = out.squeeze(-1)
                prediction = copy.deepcopy(out)
                prediction[prediction >= 0.5] = 1
                prediction[prediction < 0.5] = 0
                prediction = prediction.cuda().data.cpu().numpy()
            else:
                prediction = torch.max(out, 1)[1]
                prediction = prediction.cuda().data.cpu().numpy()

            acc_score, precision_score, recall_score, f1_score, auc_score = calc_metrics(test_y, prediction, out, args.multi_type)    
            
            acc_list.append(acc_score)
            precision_list.append(precision_score)
            recall_list.append(recall_score)
            f1_list.append(f1_score)
            if auc_score is not None:
                auc_list.append(auc_score)

    out_precision = np.mean(precision_list)
    out_recall = np.mean(recall_list)
    out_f1 = np.mean(f1_list)
    out_acc = np.mean(acc_list)
    out_auc = None if args.multi_type != 'False' else np.mean(auc_list)

    return out_precision,out_recall,out_f1,out_acc,out_auc,all_embedding

def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set log file
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # initialize data
    data = DataLoaderMUFFIN(args, logging)
    n_approved_drug = data.n_approved_drug

    # define pretrain embedding information
    structure_pre_embed = torch.tensor(data.structure_pre_embed).to(device)
    entity_pre_embed = torch.tensor(data.entity_pre_embed).to(device).float()

    all_acc_list,all_precision_list,all_recall_list,all_f1_list,all_auc_list = [],[],[],[],[]

    # train model
    # use 5-fold cross validation
    for i in range(5):

        # construct model & optimizer
        model = muffinModel(args, entity_pre_embed, structure_pre_embed)
        
        if args.use_pretrain == 2:
            # 加载模型
            model = load_model(model, args.pretrain_model_path)

        model.to(device)

        logging.info(model)

        # define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        if args.multi_type != 'False':
            print('multi-class task')
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            print('binary-class task')
            loss_func = torch.nn.BCEWithLogitsLoss()

        # Data.TensorDataset()里的两个输入是tensor类型
        train_x = torch.from_numpy(data.DDI_train_data_X[i])
        train_y = torch.from_numpy(data.DDI_train_data_Y[i])
        test_x = torch.from_numpy(data.DDI_test_data_X[i])
        test_y = torch.from_numpy(data.DDI_test_data_Y[i])

        torch_dataset_train = Data.TensorDataset(train_x, train_y)
        torch_dataset_test = Data.TensorDataset(test_x, test_y)

        loader_train = Data.DataLoader(
            dataset=torch_dataset_train,
            batch_size=args.DDI_batch_size,
            shuffle=True
        )
        loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=args.DDI_evaluate_size,
            shuffle=True
        )

        data_idx = Data.TensorDataset(torch.LongTensor(range(n_approved_drug)))
        loader_idx = Data.DataLoader(
            dataset=data_idx,
            batch_size=128,
            shuffle=False
        )

        best_epoch = -1

        epoch_list = []

        acc_list,precision_list,recall_list,f1_list,auc_list  = [],[],[],[],[]

        for epoch in range(1, args.n_epoch + 1):
            time0 = time()
            model.train()

            time1 = time()
            ddi_total_loss = 0
            n_ddi_batch = data.n_ddi_train[i] // args.DDI_batch_size + 1

            for step, (batch_x, batch_y) in enumerate(loader_train):
                iter = step + 1
                time2 = time()

                if use_cuda:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                out = model('calc_ddi_loss', batch_x, loader_idx)

                if args.multi_type == 'False':
                    out = out.squeeze(-1)
                    loss = loss_func(out, batch_y.float())
                else:
                    loss = loss_func(out, batch_y.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ddi_total_loss += loss.item()

                if (iter % args.ddi_print_every) == 0:
                    logging.info(
                        'DDI Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean '
                        'Loss {:.4f}'.format(
                            epoch, iter, n_ddi_batch, time() - time2, loss.item(), ddi_total_loss / iter))
            logging.info(
                'DDI Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                    epoch,
                    n_ddi_batch,
                    time() - time1,
                    ddi_total_loss / n_ddi_batch))

            logging.info('DDI + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

            if (epoch % args.evaluate_every) == 0:
                time3 = time()
                precision_score, recall_score, f1_score, acc_score, auc_score, all_embed = evaluate(args, model, loader_test,loader_idx)
                
                epoch_list.append(epoch)
                precision_list.append(precision_score)
                recall_list.append(recall_score)
                f1_list.append(f1_score)
                acc_list.append(acc_score)

                if args.multi_type == 'False':
                    auc_list.append(auc_score)
                    best_auc, should_stop = early_stopping(auc_list, args.stopping_steps)
                    index = auc_list.index(best_auc)
                    logging.info(
                    'DDI Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                    '{:.4f} AUC {:.4f}'.format(
                        epoch, time() - time3, precision_score, recall_score, f1_score, acc_score, auc_score))

                else:
                    best_acc, should_stop = early_stopping(acc_list, args.stopping_steps)
                    index = acc_list.index(best_acc)
                    logging.info(
                    'DDI Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                    '{:.4f}'.format(
                        epoch, time() - time3, precision_score, recall_score, f1_score, acc_score))

                if should_stop:

                    all_acc_list.append(acc_list[index])
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])
                    
                    if args.multi_type == 'False':
                        all_auc_list.append(auc_list[index])
                        logging.info('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                        '{:.4f} AUC {:.4f}'.format(precision_score, recall_score, f1_score, acc_score, auc_score))
                    else:
                        logging.info('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                        '{:.4f}'.format(precision_score, recall_score, f1_score, acc_score))

                    break

                if index == len(acc_list) - 1:
                    save_model(all_embed, model, args.save_dir, epoch, best_epoch)
                    logging.info('Save model on epoch {:04d}!'.format(epoch))
                    best_epoch = epoch

                if epoch == args.n_epoch:
                    all_acc_list.append(acc_list[index])
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])

                    if args.multi_type == 'False':
                        all_auc_list.append(auc_list[index])
                        logging.info('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                        '{:.4f} AUC {:.4f}'.format(precision_score, recall_score, f1_score, acc_score, auc_score))
                    else:
                        logging.info('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                        '{:.4f}'.format(precision_score, recall_score, f1_score, acc_score))

    print(all_acc_list)
    print(all_precision_list)
    print(all_recall_list)
    print(all_f1_list)

    mean_acc = np.mean(all_acc_list)
    mean_precision = np.mean(all_precision_list)
    mean_recall = np.mean(all_recall_list)
    mean_f1 = np.mean(all_f1_list)

    if args.multi_type == 'False':

        print(all_auc_list)
        mean_auc = np.mean(all_auc_list)
        logging.info('5-fold cross validation DDI Mean Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                     '{:.4f} AUC {:.4f}'.format(mean_precision, mean_recall, mean_f1, mean_acc, mean_auc))
    else:

        logging.info('5-fold cross validation DDI Mean Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                     '{:.4f}'.format(mean_precision, mean_recall, mean_f1, mean_acc))


if __name__ == '__main__':
    args = parse_SKGDDI_args()
    train(args)
