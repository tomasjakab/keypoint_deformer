import configargparse

from .. import datasets, models


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
        parser.add_argument('-t', '--test_config', required=False, is_config_file=True, help='config file path')
        # basic parameters
        parser.add_argument("--name", required=True, type=str, help="experiment name")
        parser.add_argument("--dataset", type=str, default='shapes', help="dataset name")
        parser.add_argument("--num_point", type=int, help="number of input points", default=1024)
        parser.add_argument("--points_dir", type=str, help="points data root", default=None)
        parser.add_argument("--dim", type=int, help="2D or 3D", default=3)
        parser.add_argument("--log_dir", type=str, help="log directory", default="./log")
        parser.add_argument("--subdir", type=str, help="save to directory name", default="test")
        parser.add_argument("--batch_size", type=int, help="batch size", default=16)
        parser.add_argument("--n_keypoints", type=int, help="")
        parser.add_argument("--cage_size", type=float, default=1.4, help="")
        parser.add_argument("--print_options", action="store_true", help="")
        # training setup
        parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
        parser.add_argument("--phase", type=str, choices=["test", "train"], default="train")
        parser.add_argument("--ckpt", type=str, help="test model")
        parser.add_argument("--seed", type=int, default=0, help="")
        parser.add_argument("--n_workers", type=int, default=8, help="")
        parser.add_argument("--iteration", type=int, default=None, help="")
        parser.add_argument("--n_iterations", type=int, default=2000, help="")
        parser.add_argument("--log_interval", type=int, default=10, help="")
        parser.add_argument("--save_interval", type=int, default=100, help="")
        # network options
        parser.add_argument("--bottleneck_size", type=int, help="bottleneck size", default=256)
        parser.add_argument("--normalization", type=str, choices=["batch", "instance", "none"], default="none")
        parser.add_argument("--disable_d_residual", dest="d_residual", action="store_false")
        parser.add_argument("--model", type=str, default='cage_skinning', help="")
        # dataset related options
        parser.add_argument("--mesh_dir", type=str, help="")
        parser.add_argument("--keypoints_dir", type=str, help="")

        self.initialized = True
        return parser


    def gather_options(self, args=None, skip_model=False, unknown_ok=False):
        # initialize parser with basic options
        if not self.initialized:
            parser = configargparse.ArgumentParser(
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        else:
            raise RuntimeError()

        # get the basic options
        opt, _ = parser.parse_known_args(args)

        if not skip_model:
            # modify model-related parser options
            model_name = opt.model
            model_option_setter = models.get_option_setter(model_name)
            parser = model_option_setter(parser)
            opt, _ = parser.parse_known_args(args)  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)
        opt, _ = parser.parse_known_args(args)  # parse again with the new defaults


        if unknown_ok:
            opt, unknown = parser.parse_known_args(args)
        else:
            opt = parser.parse_args(args)
            unknown = []

        self.parser = parser

        return opt, unknown
        

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


    def print_unknown(self, unknown):
        message = ''
        message += '----------------- Unknown options ---------------\n'
        for item in unknown:
            if item.startswith('-'):
                message += '%s, ' % item
        message += '\n'
        message += '----------------- End -------------------'
        print(message)


    def parse(self, args=None, skip_model=False, unknown_ok=False):

        opt, unknown = self.gather_options(args, skip_model=skip_model, unknown_ok=unknown_ok)
        
        if opt.print_options:
            self.print_options(opt)
        
        if unknown:
            self.print_unknown(unknown)

        if opt.phase == "test":
            assert(opt.ckpt is not None)
        opt.batch_size if opt.phase=="train" else 1
        if opt.normalization == "none":
            opt.normalization = None
        self.opt = opt
        
        return self.opt
