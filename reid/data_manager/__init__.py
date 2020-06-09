from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .campus4k_tracklet import campus4k_tracklet
from .dukemtmcreid_tracklet import dukemtmcreid_tracklet


__imgreid_factory = {
    'dukemtmcreid-tracklet': dukemtmcreid_tracklet,
}

__vidreid_factory = {
    'campus4k-tracklet': campus4k_tracklet,
}


def get_names():
    return list(__imgreid_factory.keys()) + list(__vidreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


def init_vidreid_dataset(name, **kwargs):
    if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    return __vidreid_factory[name](**kwargs)