import os
import logging
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as ttf
from torchvision import datasets, models
from termcolor import colored

from torch_randaug import RandAugment
from wideresnet import WideResNet


def get_logger(args):
    # create logger
    logger = logging.getLogger()
    # set logger level
    if args.cfg.param.logger_level == "info":
        logger.setLevel(logging.INFO)
    elif args.cfg.param.logger_level == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        raise Exception("Not supported logger level")
    # set formatting
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(colored('[%(filename)s %(lineno)d]', 'green') + \
        colored('%(levelname)s', 'blue') + colored(': %(message)s', 'yellow'))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_transforms(args):
    # strong transforms
    strong_transforms = []
    strong_transforms.append(ttf.Resize((args.cfg.transform.input_size, args.cfg.transform.input_size)))
    strong_transforms.append(ttf.ToTensor())
    strong_transforms.append(ttf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    if args.cfg.transform.strong.RA:  # Random augment + Cutout for the stong augmentation
        strong_transforms.append(RandAugment(args.cfg.transform.strong.RA_num, args.cfg.transform.strong.RA_mag))
        strong_transforms.append(Cutout())
    elif args.cfg.transform.strong.CTA:  # CTAugment including Cutout for the strong augmentation
        raise Exception("CTA currently not supported")
    # weak transforms
    weak_transforms = []
    weak_transforms.append(ttf.Resize((args.cfg.transform.input_size, args.cfg.transform.input_size)))
    weak_tf_names = args.cfg.transform.weak.augs.split(",")
    for name in weak_tf_names:
        if name == "RandomHorizontalFlip":
            weak_transforms.append(eval("ttf." + name + "(" + str(args.cfg.transform.weak.params.hflip_p) + ")"))
        elif name == "RandomAffine":
            weak_transforms.append(eval("ttf." + name + \
                "(0, translate=(" + str(args.cfg.transform.weak.params.trans_x) + "," + \
                    str(args.cfg.transform.weak.params.trans_y) + "))"))
    weak_transforms.append(ttf.ToTensor())
    weak_transforms.append(ttf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    # test transforms
    test_transforms = [ttf.Resize((args.cfg.transform.input_size, args.cfg.transform.input_size)), ttf.ToTensor(), ttf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return ttf.Compose(strong_transforms), ttf.Compose(weak_transforms), ttf.Compose(test_transforms)


def get_dataset(args):
    if args.cfg.data.name == "stl10":
        train_labeled = datasets.STL10(args.cfg.data.root, "train", transform=args.weak_transforms, download=True)
        train_unlabeled = datasets.STL10(args.cfg.data.root, "unlabeled", transform=None, download=True)
        testset = datasets.STL10(args.cfg.data.root, "test", transform=args.test_transforms, download=True)
    else:
        raise Exception("Not supported dataset")
    return train_labeled, train_unlabeled, testset


def collate_unlabeled(batch):
    pass


def get_loaders(args):
    trainloader = DataLoader(args.train_labeled, args.cfg.train.batch_size, shuffle=True, num_workers=args.cfg.train.num_workers)
    trainloader_u = DataLoader(
        args.train_unlabeled,
        args.cfg.train.batch_size * args.cfg.train.mu,
        shuffle=True,
        num_workers=args.cfg.train.num_workers,
        collate_fn=collate_unlabeled)
    testloader = DataLoader(args.testset, args.cfg.train.batch_size, shuffle=False, num_workers=args.cfg.train.num_workers)
    return trainloader, trainloader_u, testloader


def get_model(args):
    if "wideresnet" in args.cfg.model.name:
        _, depth, width = args.cfg.model.name.split("-")
        depth, width = int(depth), int(width)
        model = WideResNet(depth, args.cfg.data.num_classes, width)
    else:
        raise Exception("Not supported model architecture")
    return model


def get_optim(args):
    # optimizer
    if args.cfg.train.optim.optimizer == "SGD":
        optimizer = optim.SGD(
            args.model.parameters(),
            lr=args.cfg.train.optim.lr,
            momentum=args.cfg.train.optim.momentum,
            weight_decay=args.cfg.train.optim.weight_decay,
            nesterov=args.cfg.train.optim.nesterov)
    else:
        raise Exception("Not spported optimizer")
    # scheduler
    if args.cfg.train.optim.scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cfg.train.optim.t_max)
    else:
        raise Exception("Not spported scheduler")
    return optimizer, scheduler


def get_criterion(args):
    if args.cfg.train.criterion == "ce":
        criterion = nn.BCELoss()
    else:
        raise Exception("Not spported criterion")
    return criterion


def save_model(args):
    save_dict = {
        "state_dict": args.model.module.state_dict() if isinstance(args.model, nn.DataParallel) else args.model.state_dict(),
        "optimizer": args.optimizer.state_dict(),
        "scheduler": args.scheduler.state_dict(),
        "epoch": args.current_epoch,
        "best_val_acc": args.best_val_acc,
        "best_val_loss": args.best_val_loss
    }
    torch.save(save_dict, os.path.join(args.cfg.param.checkpoint_dir, args.cfg.param.log_name, args.cfg.param.checkpoint_name))


def load_model(args):
    cp = torch.load(os.path.join(args.cfg.param.checkpoint_dir, args.cfg.param.log_name, args.cfg.param.checkpoint_name))
    args.model.load_state_dict(cp["state_dict"])
    args.optimizer.load_state_dict(cp["optimizer"])
    args.scheduler.load_state_dict(cp["scheduler"])
    params = [cp["epoch"] + 1, cp["best_val_acc"], cp["best_val_loss"]]
    return args, params


class Cutout(torch.nn.Module):
    """
    Sets a random square patch of side-length (L×image width) pixels to gray.
    """
    def __init__(self, L=0.5):
        super().__init__()
        self.L = L
        self.sampler = torch.distributions.uniform.Uniform(0.0, L)

    def forward(self, img):
        """
        Args:
            Tensor: Tensor of size (C, H, W).
        """
        c, h, w = img.size()
        # patch size and location
        patch_size = round(w * self.sampler.sample())
        x = torch.randint(0, w - patch_size)
        y = torch.randint(0, h - patch_size)
        # gray value
        if img.dtype == torch.float32:
            if img.min >= 0:  # 0 ~ 1
                value = 0.5
            else:  # -1 ~ 1
                value = 0.0
        elif img.dtype == torch.uint8:  # 0 ~ 255
            value = 127
        else:
            raise Exception("Not supported tensor dtype.")
        # cutout
        img[:, y:y+patch_size, x:x+patch_size] = value

        return img
