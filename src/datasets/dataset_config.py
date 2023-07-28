from os.path import join

_BASE_DATA_PATH = "/datatmp/datasets/"

dataset_config = {

    'cub_200_2011_cropped': {
        'path': join(_BASE_DATA_PATH, 'cub_200_2011_cropped/'),
        'resize': (224, 224),
        'crop': None,
        'flip': False,
        'online_augment': False,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },

    'cars': {
            'path': join(_BASE_DATA_PATH, 'stanford_cars/'),
            'resize': (224, 224),
            'crop': None,
            'flip': False,
            'online_augment': False,
            'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
}

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
