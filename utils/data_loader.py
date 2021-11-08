import os
import pandas
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold

class DataLoaderMUFFIN(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name

        self.multi_type = args.multi_type

        self.entity_dim = args.entity_dim

        self.data_dir = os.path.join(args.data_dir, self.data_name)
        
        if self.multi_type == 'True':
            train_file = os.path.join(self.data_dir, 'multi_ddi_sift.txt')
        else:
            train_file = os.path.join(self.data_dir, 'DDI_pos_neg.txt')

        self.DDI_train_data_X, self.DDI_train_data_Y, self.DDI_test_data_X, self.DDI_test_data_Y = self.load_DDI_data(train_file)

        self.statistic_ddi_data()

        self.print_info(logging)

        self.train_graph = None
        
        self.load_pretrained_data()

    def load_DDI_data(self, filename):

        train_X_data = []
        train_Y_data = []
        test_X_data = []
        test_Y_data = []

        traindf = pandas.read_csv(filename, delimiter='\t', header=None)
        data = traindf.values
        DDI = data[:, 0:2]
        
        Y = data[:, 2]
        label = np.array(list(map(int, Y)))

        print('DDI data shape: {}'.format(DDI.shape))
        print('DDI label data shape: {}'.format(label.shape))

        if self.multi_type == 'True':
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=3)

        for train, test in kfold.split(DDI, label):
            train_X_data.append(DDI[train])
            train_Y_data.append(label[train])
            test_X_data.append(DDI[test])
            test_Y_data.append(label[test])

        train_X = np.array(train_X_data)
        train_Y = np.array(train_Y_data)
        test_X = np.array(test_X_data)
        test_Y = np.array(test_Y_data)

        print('Loading DDI data down!')

        return train_X, train_Y, test_X, test_Y

    # 5-fold train data length
    def statistic_ddi_data(self):
        data = []
        for i in range(len(self.DDI_train_data_X)):
            data.append(len(self.DDI_train_data_X[i]))
        self.n_ddi_train = data

    def print_info(self, logging):

        # logging.info('n_entities:         %d' % self.n_entities)
        # logging.info('n_relations:        %d' % self.n_relations)
        # logging.info('n_kg_train:         %d' % self.n_kg_train)
        logging.info('n_ddi_train:         %s' % self.n_ddi_train)

    def load_pretrained_data(self):

        print('KG-embedding loading...')
        transE_entity_path = self.args.entity_embedding_file
        transE_entity_data = np.load(transE_entity_path)

        # masking_entity_path = 'data/DRKG/gin_supervised_masking_embedding.npy'
        print('Graph-embedding loading...')
        masking_entity_path = self.args.graph_embedding_file
        masking_entity_data = np.load(masking_entity_path)            

        # apply pretrained data

        self.entity_pre_embed = transE_entity_data

        self.structure_pre_embed = masking_entity_data

        self.n_approved_drug = self.structure_pre_embed.shape[0]

        print('Loading pretrain data down!')

