from helper_tool import DataProcessing as DP
from helper_tool import Rail as cfg
from os.path import join
import numpy as np
import os, pickle



import torch.utils.data as torch_data
import torch


class raildata_RandLA(torch_data.Dataset):
    def __init__(self,mode):
        self.name = 'raildata_RandLA'
        self.dataset_path = '/home/hwq/dataset/rail_randla_0.06'
        self.label_to_names = {0:'unlabeled',
                               1:'rail',
                               2:'pole'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()]) # [0,1,2]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)} # dict {0:0,1:1,2:2}
        self.ignored_labels = np.sort([0])
        self.mode = mode

        fns = sorted(os.listdir(join(self.dataset_path,'velodyne')))
        train_index = np.load('./utils/rail_index/trainindex.npy')
        test_index = np.load('./utils/rail_index/testindex.npy')

        alldatapath=[]
        for fn in fns:
            alldatapath.append(os.path.join(self.dataset_path,fn))
        # print(alldatapath,train_index)

        self.data_list=[]
        if mode == 'training':
            for index in train_index:
                self.data_list.append(alldatapath[index])
        elif mode == 'validation':
            for index in test_index:
                self.data_list.append(alldatapath[index])
        elif mode == 'test':
            for index in test_index:
                self.data_list.append(alldatapath[index])
        self.data_list=np.asarray(self.data_list)
        self.data_list = DP.shuffle_list(self.data_list)
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        cfg.class_weights = DP.get_class_weights('Rail')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def spatially_regular_gen(self, item):
        # Generator loop

        cloud_ind = item
        pc_path = self.data_list[cloud_ind] #data_list文件名

        pc, tree, labels = self.get_data(pc_path)
        # crop a small point cloud
        pick_idx = np.random.choice(len(pc), 1) #随机选一个
        selected_pc, selected_labels, selected_idx = self.crop_pc(self.mode, pc, labels, tree, pick_idx)

        return selected_pc.astype(np.float32), selected_labels.astype(np.int32), selected_idx.astype(np.int32), np.array([cloud_ind], dtype=np.int32)

    def get_data(self, file_path):
        frame_id = file_path.split('/')[-1][:-4]
        kd_tree_path = join(self.dataset_path, 'KDTree', frame_id + '.pkl')
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        label_path = join(self.dataset_path, 'labels', frame_id + '.npy')
        labels = np.squeeze(np.load(label_path))

        return points, search_tree, labels

    @staticmethod
    def crop_pc(mode, points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1) #[1,3]
        # print('rail_dataset line 86 :',(points).shape)
        # exit()
        # cfg.num_points = 512*(points.shape[0]//512)
        kk = points.shape[0]
        # select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]  #[45056,] 数值是[0-8w多]
        if mode is 'test':
            select_idx = np.random.randint(0, kk, (512*(kk//512),))
        else:
            select_idx = np.random.randint(0,kk,(cfg.num_points,))
        select_idx = DP.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx
    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers): #4
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n) #16 return index[B,N,K]
            #取前1/4N 个点
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            #[B,N,1] 原始点集对应sub点集的idx
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)           #[N,  N/4, N/16,N/64]
            input_neighbors.append(neighbour_idx)   #[N,  N/4, N/16,N/64],k
            input_pools.append(pool_i)              #[N/4,N/16,N/64,N/256],k
            input_up_samples.append(up_i)           #[N,  N/4, N/16,N/64],1
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self,batch): #将默认的dataloader数据按collate_fn重新排序
        selected_pc, selected_labels, selected_idx, cloud_ind = [],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0]) #把每个batch的第一个取出，形成list
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)
        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = [] #input points
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()+1
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs

def main():
    from torch.utils.data import DataLoader

    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    # Create Dataset and Dataloader
    TRAIN_DATASET = raildata_RandLA(mode='training')
    VAL_DATASET = raildata_RandLA('validation')

    print(len(TRAIN_DATASET), len(VAL_DATASET))
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=5, shuffle=True, num_workers=20,
                                  worker_init_fn=my_worker_init_fn, collate_fn=TRAIN_DATASET.collate_fn)
    VAL_DATALOADER = DataLoader(VAL_DATASET, batch_size=1, shuffle=True, num_workers=20,
                                worker_init_fn=my_worker_init_fn, collate_fn=VAL_DATASET.collate_fn)

    print(len(TRAIN_DATALOADER), len(VAL_DATALOADER))
    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        points = batch_data['xyz']
        print((points[0]).shape)
        exit()


def sum_num_points():
    dataset_path = '/home/hwq/dataset/rail_randla_0.06'
    labels_path = join(dataset_path, 'labels')
    fns = sorted(os.listdir(labels_path))
    alldatapath=[]
    for fn in fns:
        alldatapath.append(os.path.join(labels_path, fn))
    print('alldata_len :',len(alldatapath))
    sum_0=[]
    sum_1=[]
    sum_2=[]
    for file_path in alldatapath:
        data = np.load(file_path)
        labels_0 = np.sum(sum(data==0))
        labels_1 = np.sum(sum(data == 1))
        labels_2 = np.sum(sum(data == 2))
        sum_0.append(labels_0)
        sum_1.append(labels_1)
        sum_2.append(labels_2)

        # print(data.shape,labels_0,labels_1,labels_2,labels_0+labels_1+labels_2)
        # exit()
    print(sum(sum_0),sum(sum_1),sum(sum_2))
if __name__ == '__main__':
    # sum_num_points()
    main()
