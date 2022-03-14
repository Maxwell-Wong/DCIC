import scanpy as sc


n_top_genes = 10000
dataset_path = './dataset/'
save_train = 'train_preprocess' + str(n_top_genes) + '.h5ad'
save_test = 'test_preprocess' + str(n_top_genes) + '.h5ad'

adata_train = sc.read_h5ad(dataset_path+'train.h5ad')
sc.pp.log1p(adata_train)
sc.pp.highly_variable_genes(adata_train,n_top_genes=n_top_genes,subset=True,inplace=True)
sc.pp.scale(adata_train,max_value=10)


adata_test = sc.read_h5ad(dataset_path+'test.h5ad')
sc.pp.log1p(adata_test)
sc.pp.highly_variable_genes(adata_test,n_top_genes=n_top_genes,subset=True,inplace=True)
sc.pp.scale(adata_test,max_value=10)
    
# 保存预处理的结果
adata_train.write(dataset_path + save_train)
adata_test.write(dataset_path + save_test)