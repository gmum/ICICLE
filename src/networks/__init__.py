from torchvision import models

# available torchvision models

allmodels = ['resnet32', 'LeNet', 'VggNet', 'protopnet_resnet34', 'protopnet_resnet50Nat', 'tesnet_resnet34']


def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == models.Inception3:
        model.head_var = 'fc'
    elif type(model) == models.ResNet:
        model.head_var = 'fc'
    elif type(model) == models.VGG:
        model.head_var = 'classifier'
    elif type(model) == models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == models.SqueezeNet:
        model.head_var = 'classifier'
    else:
        raise ModuleNotFoundError
