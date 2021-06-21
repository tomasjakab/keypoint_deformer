from ..utils import importer

MODULE = 'keypointdeformer.datasets'

def get_option_setter(dataset_name):
    return importer.get_option_setter(MODULE, dataset_name)

def get_dataset(dataset_name):
    return importer.get_model(MODULE, dataset_name)
