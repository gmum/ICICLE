import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, data, transform, class_indices=None, name='cvc'):
        """Initialization"""
        self.labels = data['y']
        if 'cvc' in name:
            self.images = [x.replace('/data/datasets/Framework', '/datatmp/datasets') for x in data['x']]
        else:
            self.images = [x for x in data['x']]
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.open(self.images[index]).convert('RGB')
        x = self.transform(x)
        y = self.labels[index]
        return x, y


def get_data(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []

    # read filenames and labels
    trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
    tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)
    psh_lines = np.loadtxt(os.path.join(path, 'push.txt'), dtype=str)
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        data[tt]['psh'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    for this_image, this_label in trn_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in psh_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['psh']['x'].append(this_image)
        data[this_task]['psh']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    import pandas as pd
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['psh']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    dict_to_df = {}
                    dict_to_df['x'] = data[tt]['trn']['x']
                    dict_to_df['y'] = data[tt]['trn']['y']
                    df_tt = pd.DataFrame.from_dict(dict_to_df)
                    df_tt['psh_name'] = df_tt['x'].apply(lambda x: x.split('/')[-1].split('.jpg')[0])
                    psh_name = data[tt]['psh']['x'][rnd_img[ii]].split('/')[-1][:-4]
                    df_to_val = df_tt[df_tt['psh_name'].str.contains(psh_name)]
                    data[tt]['val']['x'].extend(df_to_val['x'].to_list())
                    data[tt]['val']['y'].extend(df_to_val['y'].to_list())
                    to_rm = df_to_val.index.values.tolist()
                    to_rm.sort(reverse=True)
                    for idx_rm in to_rm:
                        data[tt]['trn']['x'].pop(idx_rm)
                        data[tt]['trn']['y'].pop(idx_rm)

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order
