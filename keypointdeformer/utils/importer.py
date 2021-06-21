"""
Based on CycleGAN code
"""
import importlib


def find_class_using_name(module, class_name):
    """
    module: example `deepcage.models`
    class_name: example `shapes`
    """
    # the file "module/class_name.py"
    # will be imported.
    class_filename = "." + class_name.lower()
    modellib = importlib.import_module(class_filename, module)

    # In the file, the class called ClassName() will
    # be instantiated. It is case-insensitive.
    matched_class = None
    target_class_name = class_name.replace('_', '')
    for name, found_class in modellib.__dict__.items():
        if name.lower() == target_class_name.lower():
            matched_class = found_class
    
    if matched_class is None:
        raise ImportError("In %s.py, there should be a class with a name that matches %s in lowercase." % (class_filename, target_class_name))

    return matched_class


def get_option_setter(module, class_name):
    found_class = find_class_using_name(module, class_name)
    return found_class.modify_commandline_options


def get_model(module, class_name):
    return find_class_using_name(module, class_name)
