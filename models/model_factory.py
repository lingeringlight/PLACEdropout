from models import resnet_PLACE
from models import convnet

nets_map = {
    'resnet18': resnet_PLACE.resnet18,
    'resnet50': resnet_PLACE.resnet50,
    'convnet': convnet.cnn_digitsdg,
}


def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
