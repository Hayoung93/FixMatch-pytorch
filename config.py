import argparse
from yacs.config import CfgNode as CN
from yacs.config import CfgNode


_C = CN()

_C.data = CN()
_C.data.name        = ("stl10"           , "dataset_name")
_C.data.root        = ("/data/data/stl10", "dataset_root")
_C.data.num_classes = (10                , "num_classes" )

_C.model = CN()
_C.model.name = ("wideresnet-28-2", "model_name")

_C.param = CN()
_C.param.checkpoint_dir  = ("./checkpoint/" , "checkpoint_dir" )
_C.param.log_name        = ("debug"         , "log_name"       )
_C.param.checkpoint_name = ("model_last.pth", "checkpoint_name")
_C.param.tensorboard_dir = ("./logs/"       , "tensorboard_dir")
_C.param.logger_level    = ("info"          , "logger_level"   )

_C.train = CN()
_C.train.num_epochs      = (100  , "num_epochs"     )
_C.train.batch_size      = (16   , "batch_size"     )
_C.train.criterion       = ("ce" , "criterion"      )
_C.train.skip_first_eval = (False, "skip_first_eval")
_C.train.resume          = (False, "resume"         )
_C.train.ema             = (False, "ema"            )
_C.train.unsup_weight    = (1.0  , "unsup_weight"   )
_C.train.num_workers     = (8    , "num_workers"    )
_C.train.mu              = (7    , "mu"             )
_C.train.tau             = (0.95 , "tau"            )

_C.train.optim = CN()
_C.train.optim.optimizer    = ("SGD"              , "optimizer"   )
_C.train.optim.scheduler    = ("CosineAnnealingLR", "scheduler"   )
_C.train.optim.lr           = (0.005              , "lr"          )
_C.train.optim.weight_decay = (0.0005             , "weight_decay")
_C.train.optim.momentum     = (0.9                , "momentum"    )
_C.train.optim.nesterov     = (False              , "nesterov"    )
_C.train.optim.t_max        = (100                , "t_max"       )

_C.transform = CN()
_C.transform.input_size = (224  , "input_size")

_C.transform.strong = CN()
_C.transform.strong.RA         = (False    , "randaug"    )
_C.transform.strong.RA_num     = (2        , "ra_num"     )
_C.transform.strong.RA_mag     = (1        , "ra_mag"     ) 
_C.transform.strong.CTA        = (False    , "ctaug"      )

_C.transform.weak = CN()
_C.transform.weak.augs   = ("RandomHorizontalFlip,RandomAffine", "weak_augs"  )

_C.transform.weak.params = CN()
_C.transform.weak.params.hflip_p = (0.5  , "hflip_p")
_C.transform.weak.params.trans_x = (0.125, "trans_x")
_C.transform.weak.params.trans_y = (0.125, "trans_y")


def get_cfg_defaults():
    return _C.clone()


def merge_args(args, cfg):
    new_cfg = CfgNode({})
    new_cfg.set_new_allowed(True)
    key_gen = get_cfg_keys(cfg, cfg.keys())
    while True:
        try:
            key = next(key_gen)
            value, args_key = eval("cfg." + key)
            if (args_key in args) and (eval("args." + args_key) is not None):
                value = eval("args." + args_key)
            key_split =  key.split(".")
            t1 = {key_split[-1]: value}
            t2 = {}
            for k in key_split[:-1][::-1]:
                if t1 == {}:
                    t1[k] = t2
                    t2 = {}
                else:
                    t2[k] = t1
                    t1 = {}
            if t1 == {}:
                t2 = CfgNode(t2)
                new_cfg = merge_cfg(t2, new_cfg)
            else:
                t1 = CfgNode(t1)
                new_cfg = merge_cfg(t1, new_cfg)
        except StopIteration:
            break
    return new_cfg


def get_cfg_keys(cn, keys):
    for key in keys:
        cur_node = eval("cn." + key)
        if type(cur_node) == CfgNode:
            yield from get_cfg_keys(cn, list(map(lambda x: key + "." + x, cur_node.keys())))
        else:
            yield key


def merge_cfg(a, b):
    for k, v in a.items():
        if k in b:
            if isinstance(v, CfgNode):
                merge_cfg(v, b[k])
            else:
                b[k] = v
        else:
            b[k] = v
    return b


def get_args():
    parser = argparse.ArgumentParser(description='See config.py for detailed information')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--log_name", type=str)
    parser.add_argument("--checkpoint_name", type=str)
    parser.add_argument("--tensorboard_dir", type=str)
    parser.add_argument("--logger_level", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--criterion", type=str)
    parser.add_argument("--skip_first_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--unsup_weight", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--mu", type=int)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--t_max", type=int)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--randaug", action="store_true")
    parser.add_argument("--ra_num", type=int)
    parser.add_argument("--ra_mag", type=int)
    parser.add_argument("--ctaug", action="store_true")
    parser.add_argument("--weak_augs", type=str)
    parser.add_argument("--hflip_p", type=float)
    parser.add_argument("--trans_x", type=float)
    parser.add_argument("--trans_y", type=float)

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg = merge_args(args, cfg)
    return cfg
