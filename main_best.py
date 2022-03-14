import argparse
from cgi import test
import re
from xml import dom
import pandas as pd
import anndata as ad
from scAdapt_best import scAdapt
import os
import numpy as np
import scanpy as sc


def downsampling(adata, cell_type, sample_num):
    new_adata = adata[adata.obs['CellType1'] == cell_type]
    random_index = np.random.choice(new_adata.shape[0], sample_num)
    new_adata = new_adata[random_index]
    return new_adata

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

    if args.mode == 0:
        adata = sc.read_h5ad(dataset_path+'train_preprocess.h5ad')
        adata_list = []
        for cell_type in cell_label_digit:
            adata_list.append(downsampling(adata, cell_type, args.sample_num))
        
        adata = ad.concat([adata_list[i] for i in range(len(cell_label_digit))])
        train_ratio=0.8
        num_trains = int(adata.shape[0] * train_ratio)
        rand_index = np.random.choice(adata.shape[0], adata.shape[0], replace=False)
        train_index, test_index = rand_index[:num_trains], rand_index[num_trains:]

        adata_train = adata[train_index]
        adata_test = adata[test_index]

        print(adata_train)
        print(adata_test)

        normcounts = ad.concat([adata_train,adata_test]).X

        # labels
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

        domain_labels = np.array(train_labels_domain)  # 标记train和test各有多少
        print("domain_labels_pre:",domain_labels)

        data_set = {'features': normcounts, 'train_labels':train_labels_digit_np,'test_labels':test_labels_digit_np,
               'accessions':domain_labels,'barcode':adata_test.obs['barcode']}

        return data_set
    
    elif args.mode == 1:

        adata_train = sc.read_h5ad(dataset_path+'train_preprocess.h5ad')
        adata_test = sc.read_h5ad(dataset_path+'test_preprocess.h5ad')

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

        domain_labels = np.array(train_labels_domain)  # 标记train和test各有多少
        print("domain_labels_pre:",domain_labels)

        data_set = {'features': normcounts, 'train_labels':train_labels_digit_np,'test_labels':test_labels_digit_np,
               'accessions':domain_labels,'barcode':adata_test.obs['barcode']}
        return data_set
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scAdapt: virtual adversarial domain adaptation network')
    parser.add_argument('--method', type=str, default='DANN', choices=['DANN', 'mmd'])
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding_size')
    parser.add_argument('--source_name', type=str, default='train')
    parser.add_argument('--target_name', type=str, default='test')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--BNM_coeff', type=float, default=0.2, help="regularization coefficient for BNM loss")
    parser.add_argument('--centerloss_coeff', type=float, default=1.0,  help='regularization coefficient for center loss')
    parser.add_argument('--semantic_coeff', type=float, default=1.0, help='regularization coefficient for semantic loss')
    parser.add_argument('--DA_coeff', type=float, default=1.0, help="regularization coefficient for domain alignment loss")
    parser.add_argument('--pseudo_th', type=float, default=0.0, help='pseudo_th')
    parser.add_argument('--cell_th', type=int, default=20, help='cell_th')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient for VAT loss')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT')


    parser.add_argument('--num_iterations', type=int, default=35010, help="num_iterations")
    parser.add_argument('--epoch_th', type=int, default=120000, help='epoch_th')
    parser.add_argument('--epoch_per', type=int, default=2800, help='epoch_per')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--mode', type=int, default=0, help="zero means test, one means submit")
    parser.add_argument('--sample_num', type=int, default=20000, help="number of samples per cell type")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  #'0,1,2,3'
    print(args)
    data_set = preprocess(args)
    scAdapt(args, data_set=data_set)


