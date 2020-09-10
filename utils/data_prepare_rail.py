import pickle, yaml, os, sys
import numpy as np
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_tool import DataProcessing as DP

grid_size = 0.06

dataset_path = '/home/hwq/dataset/labeled/label2'
output_path = '/home/hwq/dataset/rail_randla' + '_' + str(grid_size)



# seq_path = dataset_path
seq_path_out = output_path
pc_path = dataset_path
pc_path_out = join(seq_path_out, 'velodyne')
KDTree_path_out = join(seq_path_out, 'KDTree')
os.makedirs(seq_path_out) if not exists(seq_path_out) else None
os.makedirs(pc_path_out) if not exists(pc_path_out) else None
os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None

# label_path = seq_path
label_path_out = join(seq_path_out, 'labels')
os.makedirs(label_path_out) if not exists(label_path_out) else None
data_list = np.sort(os.listdir(dataset_path))
print(' start',len(data_list))
# scan_list = np.sort(os.listdir(pc_path))
len_points=100000
len_sub=100000
for i,data_id in enumerate(data_list):
    # if i >20:
    #     exit()
    print(join(dataset_path,data_id))
    data = np.load(join(dataset_path,data_id))
    points = data[:,:3].astype(np.float32)
    labels = data[:,-1].astype(np.int32)

    sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
    print('test',points.shape,labels.shape,sub_points.shape, sub_labels.shape)
    # exit()

    if len_points>points.shape[0]:
        len_points = points.shape[0]
    if len_sub>sub_points.shape[0]:
        len_sub =  sub_points.shape[0]
    search_tree = KDTree(sub_points)
    KDTree_save = join(KDTree_path_out, str(data_id[:-4]) + '.pkl')
    np.save(join(pc_path_out, data_id)[:-4], sub_points)
    np.save(join(label_path_out, data_id)[:-4], sub_labels)
    with open(KDTree_save, 'wb') as f:
        pickle.dump(search_tree, f)

print(len_points,len_sub)