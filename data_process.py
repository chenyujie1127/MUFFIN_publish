import pandas as pd
import re
import random
from argparse import ArgumentParser

parser = ArgumentParser("Data Processing")
parser.add_argument('-s', '--smiles', type=str,default='./data/DRKG/drugname_smiles.txt',
                    help="Path to the file of SMILES")
parser.add_argument('-kg', '--know', type=str,default='./data/DRKG/drkg.tsv',
                    help="Path to the file of KG")
parser.add_argument('-mc', '--multi_class', type=str, default='./data/DRKG/multi_ddi_name.txt',
                    help='Path to the multi-class file')
args = parser.parse_args()


## Dataset Preparation

"""

##############   What You Need Prepare   ###################

Before you run this code, you need prepare files listing below:

[1] smiles_file: ./data/DRKG/drugname_smiles.txt
[2] drkg_file: ./data/DRKG/drkg.tsv
[3] multi_ddi_ori_file: ./data/DRKG/multi_ddi_name.txt

"""

############################### Step1:  Prepare Approved Drug SMILES File  ###############################

"""
this file looks like "Compound::DB00119	CC(=O)C(O)=O"
if your drug name not this form, just map your name into the DRKG-style name form
maybe you just need to append "Compound::" before the "DB..."(DrugBank form) name
"""

# smiles_file = './data/DRKG/drugname_smiles.txt'
smiles_file = args.smiles

############### Step2:  Download KG dataset & Prepare binary-class DDI dataset ############################

"""
Step2.1: run the code in your terminal:

wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz

When you get "drkg.tsv" file, put it into the "./data/DRKG" directory. 

"""

# drkg_file = './data/DRKG/drkg.tsv'
drkg_file = args.know

"""
Step2.2: delete DDI dataset from Knowledge Graph as new KG.
"""

def delete_drugbank_hetionet_ddi_from_drkg(infile, outfile):
    train = []
    df = pd.read_csv(infile, sep="\t", header=None)
    triples = df.values.tolist()
    print('Original KG length: {}'.format(len(triples)))

    for i in range(len(triples)):
        drug_1, relation, drug_2 = triples[i][0], triples[i][1], triples[i][2]
        # result = re.match(r'DRUGBANK::', relation)
        # result2 = re.match(r'Hetionet::', relation)
        if relation not in ['DRUGBANK::ddi-interactor-in::Compound:Compound','Hetionet::CrC::Compound:Compound']:
            l1 = "{}{}{}{}{}\n".format(drug_1, '\t', relation, '\t', drug_2)
            # print(l1)
            train.append(l1)

    with open(outfile, "w+") as f:
        f.writelines(train)
    n_kg = len(train)
    # 4488504个三元组
    print('New triples length: {}'.format(n_kg))
    print('Generate triple files without DDI pairs successfully!')

new_drkg_file = './data/DRKG/train_without_ddi_raw.tsv'
delete_drugbank_hetionet_ddi_from_drkg(drkg_file, new_drkg_file)

"""
Step2.3: Map Entity ID & Generate entity/relation/triples file & Generate binary-class positive data.
"""
def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def generate_entity_relation_id_file(delimiter, smilesfile, new_drkg_file, drkg_file, entity_file, relation_file, triple_file,
                                     ddi_name_file, ddi_id_file):
    entity_map = {}
    rel_map = {}
    train = []

    smiles_file = open(smilesfile, 'r')
    approved_drug_list = set()
    for line in smiles_file :
        drug = line.replace('\n', '').replace('\r', '').split('\t')
        # approved_drug_list.append(drug[0])
        approved_drug_list.add(drug[0])
        drug_id = _get_id(entity_map, drug[0])
        # print(drug, drug_id)

    df = pd.read_csv(new_drkg_file, sep="\t", header=None)
    triples = df.values.tolist()

    for i in range(len(triples)):
        src, rel, dst = triples[i][0], triples[i][1], triples[i][2]
        src_id = _get_id(entity_map, src)
        dst_id = _get_id(entity_map, dst)
        rel_id = _get_id(rel_map, rel)
        train_id = "{}{}{}{}{}\n".format(src_id, delimiter, rel_id, delimiter, dst_id)
        # print(train_id)
        train.append(train_id)

    entities = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(entity_map.items(), key=lambda x: x[1])]
    with open(entity_file, "w+") as f:
        f.writelines(entities)
    n_entities = len(entities)

    relations = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(rel_map.items(), key=lambda x: x[1])]
    with open(relation_file, "w+") as f:
        f.writelines(relations)
    n_relations = len(relations)

    with open(triple_file, "w+") as f:
        f.writelines(train)
    n_kg = len(train)

    # the code down from here is just extract DDI pairs from DRKG and transfer it into id form.
    df_2 = pd.read_csv(drkg_file, sep="\t", header=None)
    triples2 = df_2.values.tolist()

    ddi_name = []
    ddi = []
    ddi_name_list = set()
    for i in range(len(triples2)):
        src, rel, dst = triples2[i][0], triples2[i][1], triples2[i][2]
        if rel in ['DRUGBANK::ddi-interactor-in::Compound:Compound', 'Hetionet::CrC::Compound:Compound']:
            # 存储有SMILES的DDI信息
            if src in approved_drug_list and dst in approved_drug_list:
                ddi_pair_single = "{}{}{}\n".format(src, '\t', dst)
                ddi_pair_single_reverse = "{}{}{}\n".format(dst, '\t', src)
                # print(ddi_pair_single)
                if ddi_pair_single not in ddi_name_list:
                    ddi_name_list.add(ddi_pair_single)
                    ddi_name.append(ddi_pair_single)
                    drug_id_1 = _get_id(entity_map, src)
                    drug_id_2 = _get_id(entity_map, dst)
                    ddi_id = "{}{}{}\n".format(drug_id_1, delimiter, drug_id_2)
                    ddi.append(ddi_id)
                # else:
                #     print('positive pair replicate: {}'.format(ddi_pair_single))
                    
                if ddi_pair_single_reverse not in ddi_name_list:
                    ddi_name_list.add(ddi_pair_single_reverse)
                    ddi_name.append(ddi_pair_single_reverse)
                    drug_id_1 = _get_id(entity_map, src)
                    drug_id_2 = _get_id(entity_map, dst)
                    ddi_id_reverse = "{}{}{}\n".format(drug_id_2, delimiter, drug_id_1)
                    ddi.append(ddi_id_reverse)
                # else:
                #     print('reverse pair replicate: {}'.format(ddi_pair_single_reverse))

    with open(ddi_name_file, "w+") as f:
        f.writelines(ddi_name)

    with open(ddi_id_file, 'w+') as f:
        f.writelines(ddi)

    n_ddi = len(ddi)

    print('entity length: {}'.format(n_entities))
    print('relation length: {}'.format(n_relations))
    print('triples length: {}'.format(n_kg))
    print('binary ddi length: {}'.format(n_ddi))
    print('You have done it successfully!')


entity_file = './data/DRKG/entities.tsv'
relation_file = './data/DRKG/relations.tsv'
triple_file = './data/DRKG/train.tsv'

ddi_pos_file = './data/DRKG/ddi_facts_pos.txt'
ddi_pos_id_file = './data/DRKG/ddi_facts_pos_id.txt'

generate_entity_relation_id_file('\t', smiles_file, new_drkg_file, drkg_file, entity_file, relation_file,
                                     triple_file, ddi_pos_file, ddi_pos_id_file)

"""
Step2.4: Generate binary-class negtive data & Generate full binary-class dataset.
"""

def generate_positive_pairs(DDI_positive_file):
    druglist_left = []
    druglist_right = []
    DDI = {}
    DDI_pos_num = 0
    with open(DDI_positive_file, 'r') as csvfile:
        for row in csvfile:
            DDI_pos_num += 1
            lines = row.replace('\n', '').replace('\r', '').split('\t')
            drug_1 = lines[0]
            drug_2 = lines[1]
            if drug_1 not in DDI:
                DDI[drug_1] = []
            DDI[drug_1] += [drug_2]
            druglist_left.append(drug_1)
            druglist_right.append(drug_2)
        druglist_left = list(set(druglist_left))
        druglist_right = list(set(druglist_right))
    print('generate_positive_pairs.')
    return druglist_left, druglist_right, DDI, DDI_pos_num


def generate_negative_pairs(DDI_pos, druglist_right):
    DDI_neg = {}
    druglist_neg = druglist_right[:]
    for k, v in DDI_pos.items():
        if k not in DDI_neg:
            DDI_neg[k] = []
        if k in druglist_neg:
            druglist_neg.remove(k)
        for i in v:
            if i in druglist_neg:
                druglist_neg.remove(i)
        DDI_neg[k] = druglist_neg
        druglist_neg = druglist_right[:]
    print('generate_negative_pairs.')
    return DDI_neg


def generate_neg_data(DDI_pos_num, DDI_neg, output_negative):
    DDI_pos_neg = dict()
    drug1_name = 'Drug_1'
    drug2_name = 'Drug_2'
    drug_index = 0
    c = 0

    for drug_1, v in DDI_neg.items():
        if drug_1 not in DDI_pos_neg:
            DDI_pos_neg[drug_index] = dict()
        for drug_2 in v:
            DDI_pos_neg[drug_index] = dict()
            DDI_pos_neg[drug_index][drug1_name] = drug_1
            DDI_pos_neg[drug_index][drug2_name] = drug_2
            drug_index += 1

    resultList = random.sample(range(0, drug_index), DDI_pos_num)
    for i in resultList:
        drug_1_id = DDI_pos_neg[i][drug1_name]
        drug_2_id = DDI_pos_neg[i][drug2_name]
        c += 1
        outline = drug_1_id + '\t' + drug_2_id + '\t' + str(0) + '\n'
        output_negative.write(outline)
    output_negative.close()
    # print(c)
    print('Yep! Finish generate_neg_data_file.')


def concate_pos_neg_data(infile_1, infile_2, outputfile):
    c = 0
    for line in infile_1:
        c += 1
        lines = line.replace('\n', '').replace('\r', '').split('\t')
        drug_1 = lines[0]
        drug_2 = lines[1]
        drug_state = 1
        outline = "{}\t{}\t{}\n".format(drug_1, drug_2, drug_state)
        # outline = drug_1 + '\t' + drug_2 + '\t' + drug_state + '\n'
        outputfile.write(outline)
    print('pos data finish!')
    for line in infile_2:
        c += 1
        lines = line.replace('\n', '').replace('\r', '').split('\t')
        drug_1 = lines[0]
        drug_2 = lines[1]
        drug_state = int(lines[2])
        outline = "{}\t{}\t{}\n".format(drug_1, drug_2, drug_state)
        # outline = drug_1 + '\t' + drug_2 + '\t' + drug_state + '\n'
        outputfile.write(outline)
    print('neg data finish!')
    outputfile.close()
    print('binary DDI datasets length: {}'.format(c))
    print('DDI_pos_neg.txt done!')

DDI_negfile = './data/DRKG/DDI_neg.txt'
DDI_outfile = './data/DRKG/DDI_pos_neg.txt'

drug_list_left, drug_list_right, DDI_positive, DDI_positive_num = generate_positive_pairs(ddi_pos_id_file)
DDI_negative = generate_negative_pairs(DDI_positive, drug_list_right)
generate_neg_data(DDI_positive_num, DDI_negative, open(DDI_negfile, 'w'))
concate_pos_neg_data(open(ddi_pos_id_file, 'r'), open(DDI_negfile, 'r'), open(DDI_outfile, 'w'))


################################## Step3:  Prepare multi-class DDI dataset ################################

# multi_ddi_ori_file = './data/DRKG/multi_ddi_name.txt' 
multi_ddi_ori_file = args.multi_class
multi_ddi_id_file = './data/DRKG/multi_ddi_sift.txt'

def generate_multi_ddi_id_file(multi_ddi_name,entityfile,multi_ddi_id):

    df = pd.read_csv(entityfile, sep="\t", header=None)
    entities = df.values.tolist()[:2322]
    dic = {}
    for id,drug in entities:
        dic[drug] = id

    data = []
    infile = open(multi_ddi_name, 'r')
    for line in infile:
        drug1,drug2,ddi_class = line.replace('\n', '').replace('\r', '').split('\t')
        if drug1 in dic and drug2 in dic:
            drug1_id,drug2_id = dic[drug1],dic[drug2]
            l1 = "{}{}{}{}{}\n".format(drug1_id, '\t', drug2_id,'\t', int(ddi_class))
            data.append(l1)

    with open(multi_ddi_id, "w+") as f:
        f.writelines(data)

    print('Yep ~ Multi-class data finish!')

generate_multi_ddi_id_file(multi_ddi_ori_file,entity_file,multi_ddi_id_file)




