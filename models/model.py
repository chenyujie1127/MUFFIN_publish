import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_INIT_EPS = 2.0
gamma = 12.0

class muffinModel(nn.Module):

    def __init__(self, args, entity_pre_embed=None, structure_pre_embed=None):

        super(muffinModel, self).__init__()

        # concat | sum | double [cross & scalar-level]
        self.fusion_type = args.feature_fusion

        # embedding setting
        self.structure_dim = args.structure_dim
        self.entity_dim = args.entity_dim

        # embedding data
        self.structure_pre_embed = structure_pre_embed
        self.entity_pre_embed = entity_pre_embed
        self.n_approved_drug = structure_pre_embed.shape[0]

        # self.n_entities = n_entities
        # self.n_relations = n_relations

        self.multi_type = args.multi_type

        # self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)
        # self.mess_dropout = eval(args.mess_dropout)
        # self.n_layers = len(eval(args.conv_dim_list))

        # self.ddi_l2loss_lambda = args.DDI_l2loss_lambda

        self.hidden_dim = args.entity_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / self.hidden_dim

        # fusion type
        if self.fusion_type == 'concat':

            self.layer1_f = nn.Sequential(nn.Linear(self.structure_dim + self.entity_dim, self.entity_dim),
                                          nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))
            self.layer2_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim), 
                                          nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))
            self.layer3_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim), 
                                          nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))

        elif self.fusion_type == 'sum':

            self.W_s = nn.Linear(self.structure_dim, self.entity_dim)
            self.W_e = nn.Linear(self.entity_dim, self.entity_dim)

        elif self.fusion_type == 'double':

            self.druglayer_structure = nn.Linear(self.structure_dim, self.entity_dim)
            self.druglayer_KG = nn.Linear(self.entity_dim, self.entity_dim)

            self.add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.cross_add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.multi_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.activate = nn.ReLU()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=args.in_channels, out_channels=args.out_channels, kernel_size=(args.kernel, args.kernel)),
                nn.BatchNorm2d(args.out_channels), nn.MaxPool2d((args.pooling_size, args.pooling_size)), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=args.out_channels, out_channels=args.out_channels, kernel_size=(args.kernel, args.kernel)),
                nn.BatchNorm2d(args.out_channels), nn.MaxPool2d((args.pooling_size, args.pooling_size)), nn.ReLU())
            
            self.conv1_out = (self.entity_dim - args.kernel + 1)/2
            self.conv2_out = (self.conv1_out - args.kernel + 1)/2

            self.fc1 = nn.Sequential(nn.Linear(int(self.conv2_out * self.conv2_out * args.out_channels), self.entity_dim), 
                                     nn.BatchNorm1d(self.entity_dim),
                                     nn.ReLU(True))

            self.fc2_global = nn.Sequential(
                nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
                nn.ReLU(True))
            self.fc2_global_reverse = nn.Sequential(
                nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
                nn.ReLU(True))
            self.fc2_cross = nn.Sequential(
                nn.Linear(self.entity_dim * 4, self.entity_dim),
                nn.ReLU(True))

        if self.fusion_type in ['double']:
            self.all_embedding_dim = (self.entity_dim * 3 + self.structure_dim + self.entity_dim) * 2
        elif self.fusion_type in ['concat','sum']:
            self.all_embedding_dim = self.entity_dim

        self.layer1 = nn.Sequential(nn.Linear(self.all_embedding_dim, args.n_hidden_1), nn.BatchNorm1d(args.n_hidden_1),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(args.n_hidden_1, args.n_hidden_2), nn.BatchNorm1d(args.n_hidden_2),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(args.n_hidden_2, args.out_dim))

    def generate_fusion_feature(self, train_data,batch_data):
        # we focus on approved drug
        global embedding_data
        global embedding_data_reverse

        self.entity_embed_pre = self.entity_pre_embed[:self.n_approved_drug, :]

        if self.fusion_type == 'concat':

            x = torch.cat([self.structure_pre_embed, self.entity_embed_pre], dim=1)
            x = self.layer1_f(x)
            x = self.layer2_f(x)
            x = self.layer3_f(x)

            return x

        elif self.fusion_type == 'sum':

            structure = self.W_s(self.structure_pre_embed)
            entity = self.W_e(self.entity_embed_pre)
            add_structure_entity = structure + entity

            return add_structure_entity

        elif self.fusion_type == 'double':

            structure = self.druglayer_structure(self.structure_pre_embed)
            entity = self.druglayer_KG(self.entity_embed_pre)

            structure_embed_reshape = structure.unsqueeze(-1)  # batch_size * embed_dim * 1
            entity_embed_reshape = entity.unsqueeze(-1)  # batch_size * embed_dim * 1

            entity_matrix = structure_embed_reshape * entity_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim
            entity_matrix_reverse = entity_embed_reshape * structure_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim

            entity_global = entity_matrix.view(entity_matrix.size(0), -1)
            entity_global_reverse = entity_matrix_reverse.view(entity_matrix.size(0), -1)

            entity_matrix_reshape = entity_matrix.unsqueeze(1)
            entity_matrix_reshape_reverse = entity_matrix_reverse.unsqueeze(1)

            for i, data in enumerate(batch_data):

                entity_matrix_reshape = entity_matrix_reshape.to('cuda')
                entity_data = entity_matrix_reshape.index_select(0, data[0].to('cuda'))

                entity_matrix_reshape_reverse = entity_matrix_reshape_reverse.to('cuda')
                entity_reverse = entity_matrix_reshape_reverse.index_select(0, data[0].to('cuda'))

                out = self.conv1(entity_data)
                out = self.conv2(out)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)

                out2 = self.conv1(entity_reverse)
                out2 = self.conv2(out2)
                out2 = out2.view(out2.size(0), -1)
                out2 = self.fc1(out2)

                if i == 0:
                    embedding_data = out
                    embedding_data_reverse = out2
                else:
                    embedding_data = torch.cat((embedding_data, out), 0)
                    embedding_data_reverse = torch.cat((embedding_data_reverse, out2), 0)

            global_local_before = torch.cat((embedding_data, entity_global), 1)
            cross_embedding_pre = self.fc2_global(global_local_before)

            global_local_before_reverse = torch.cat((embedding_data_reverse, entity_global_reverse), 1)
            cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)

            out3 = self.activate(self.multi_drug(structure * entity))

            out_concat = torch.cat(
                (self.structure_pre_embed, self.entity_embed_pre, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)

            return out_concat

    def train_DDI_data(self, train_data,batch_data):

        # all_embed = self.generate_fusion_feature(batch_data)

        drug1_embed = self.all_embed[train_data[:, 0]]
        drug2_embed = self.all_embed[train_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def test_DDI_data(self, test_data, batch_data):

        # all_embed = self.generate_fusion_feature(batch_data)
        drug1_embed = self.all_embed[test_data[:, 0]]
        drug2_embed = self.all_embed[test_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.multi_type != 'False':
            pred = F.softmax(x, dim=1)
        else:
            pred = torch.sigmoid(x)

        return pred, self.all_embed

    def forward(self, mode, *input):
        self.all_embed = self.generate_fusion_feature(*input)
        if mode == 'calc_ddi_loss':
            return self.train_DDI_data(*input)
        if mode == 'predict':
            return self.test_DDI_data(*input)
