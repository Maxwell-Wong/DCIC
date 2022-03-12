import argparse
from cgi import test
from xml import dom
import pandas as pd
import anndata as ad
from scAdapt import scAdapt
import os
import numpy as np
import scanpy as sc
def downsampling(adata,cell_type):  # 下采样处理
    train_list = adata.obs['CellType1'][adata.obs['CellType1']==cell_type].tolist()
    train_index =[]
    for i, _ in enumerate(train_list):
        train_index.append(i)
    train_index = np.asarray(train_index)
    train_idx = np.random.choice(train_index,2000)
    new_train = []
    for i in train_idx:
        new_train.append(adata.X[i])
    new_train = np.asarray(new_train)
    return new_train

def preprocess(args):

    cell_label_digit = {
        'Adipocyte' :0, 'Cardiomyocyte cell':1,
        'Endothelial cell':2,  'Fibroblast':3,
        'Lymphoid cell':4, 'Mesothelial cell' :5,
        'Myeloid cell' :6, 'Pericyte':7,
        'Smooth muscle cell':8
    }

    dataset_path = args.dataset_path  #"../processed_data/"
    print("dataset_path: ", dataset_path)

    n_top_genes = 5700
    adata_train = sc.read_h5ad(dataset_path+'train.h5ad')
    # import pdb; pdb.set_trace()
    sc.pp.log1p(adata_train)
    sc.pp.highly_variable_genes(adata_train,n_top_genes=n_top_genes,subset=True,inplace=True)
    sc.pp.scale(adata_train,max_value=10)


    adata_test = sc.read_h5ad(dataset_path+'test.h5ad')
    sc.pp.log1p(adata_test)
    sc.pp.highly_variable_genes(adata_test,n_top_genes=n_top_genes,subset=True,inplace=True)
    sc.pp.scale(adata_test,max_value=10)
    
     # 对所有细胞类型下采样
    # new_train = []
    # for cell_type in cell_label_digit:
    #     train_T = downsampling(adata_train, cell_type)
    #     new_train.append(train_T)
    # new_train = np.asarray(new_train)
    # new_train = new_train.reshape(-1, n_top_genes)

    # new_test = []
    # for cell_type in cell_label_digit:
    #     test_T = downsampling(adata_test, cell_type)
    #     new_test.append(test_T)
    # new_test = np.asarray(new_test)
    # new_test = new_test.reshape(-1, n_top_genes)
    # normcounts = np.append(new_train, new_test,axis=0)
    # normcounts = normcounts.reshape(-1,n_top_genes)

    normcounts = ad.concat([adata_train,adata_test]).X

    #labels
    train_labels = adata_train.obs['CellType1'].values
    test_labels = adata_test.obs['CellType1'].values

    # transfer to digit
    train_labels_digit = [cell_label_digit[labels] for labels in train_labels]
    test_labels_digit = [cell_label_digit[labels] for labels in test_labels]

    train_labels_digit_np = np.array(train_labels_digit)
    test_labels_digit_np = np.array(test_labels_digit)


    #domain labels
    test_labels_domain = ['test']*adata_test.shape[0]
    train_labels_domain= ['train']*adata_train.shape[0]

    train_labels_domain.extend(test_labels_domain)

    # test_labels_domain =  ['test']*new_train.shape[0]
    # train_labels_domain = ['train']*new_test.shape[0]
    # train_labels_domain.extend(test_labels_domain)
    domain_labels = np.array(train_labels_domain)  # 标记train和test各有多少
    print("domain_labels_pre:",domain_labels)


    data_set = {'features': normcounts, 'train_labels':train_labels_digit_np,'test_labels':test_labels_digit_np,
               'accessions':domain_labels,'barcode':adata_test.obs['barcode']}

    # import pdb; pdb.set_trace()
    return data_set
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scAdapt: virtual adversarial domain adaptation network')
    parser.add_argument('--method', type=str, default='DANN', choices=['DANN', 'mmd'])
    parser.add_argument('--batch_size', type=int, default=900, help='batch_size')
    parser.add_argument('--embedding_size', type=int, default=900, help='embedding_size')
    parser.add_argument('--source_name', type=str, default='train')
    parser.add_argument('--target_name', type=str, default='test')
    parser.add_argument('--result_path', type=str, default='./result_singlecell3/')
    parser.add_argument('--dataset_path', type=str, default='/home/hyangjy/scAdapt/dataset/')
    parser.add_argument('--num_iterations', type=int, default=13010, help="num_iterations")
    parser.add_argument('--BNM_coeff', type=float, default=0.2, help="regularization coefficient for BNM loss")
    parser.add_argument('--centerloss_coeff', type=float, default=1.0,  help='regularization coefficient for center loss')
    parser.add_argument('--semantic_coeff', type=float, default=1.0, help='regularization coefficient for semantic loss')
    parser.add_argument('--DA_coeff', type=float, default=1.0, help="regularization coefficient for domain alignment loss")
    parser.add_argument('--pseudo_th', type=float, default=0.0, help='pseudo_th')
    parser.add_argument('--cell_th', type=int, default=20, help='cell_th')
    parser.add_argument('--epoch_th', type=int, default=14000, help='epoch_th')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient for VAT loss')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  #'0,1,2,3'
    print(args)
    data_set = preprocess(args)
    scAdapt(args, data_set=data_set)


