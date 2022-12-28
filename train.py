import torch
import os
from argparse import Namespace
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import get_args
from utils import get_logger, get_transforms, get_dataset, get_loaders, get_model, get_optim, get_criterion, save_model, load_model


def train(args):
    model = args.model
    model.train()
    return args


def eval(args):
    model = args.model
    model.eval()
    running_acc = 0.0
    running_loss = 0.0
    num_samples = 0
    args.val_acc = 0.0
    args.val_loss = 0.0

    pbar = tqdm(args.testloader)
    for i, (inputs, labels) in enumerate(pbar):
        num_samples += inputs.shape[0]
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        # forward
        outputs = model(inputs).softmax(1)
        # compute acc & loss
        running_acc = (outputs.argmax(1) == labels).sum().item()
        args.val_acc += running_acc
        loss = args.criterion(outputs, torch.nn.functional.one_hot(labels, 10).float())
        running_loss = loss.item()
        args.val_loss += running_loss * inputs.shape[0]
        # set tqdm description
        pbar.set_description("Running Acc:{:.4f}, Running Loss:{:.4f}, Avg Acc:{:.4f}, Avf Loss:{:.4f}".format(
            running_acc, running_loss, args.val_acc / num_samples, args.val_loss / num_samples))
    # average val acc & loss
    args.val_acc = args.val_acc / num_samples
    args.val_loss = running_loss / num_samples
    # update best val acc
    if args.val_acc > args.best_val_acc:
        args.best_val_acc = args.val_acc
    # update best val loss
    if args.val_loss < args.best_val_loss:
        args.best_val_loss = args.val_loss
    return args


def main(args):
    # tensorboard setting
    tensorboard_path = os.path.join(args.cfg.param.tensorboard_dir, args.cfg.param.log_name)
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)
    args.writer = writer

    # logger setting
    logger = get_logger(args)
    args.logger = logger
    args.logger.info(f"Config: {args.cfg}")
    args.logger.info(f"Device: {args.device}")

    # transforms
    strong_transforms, weak_transforms, test_transforms = get_transforms(args)
    args.logger.info(f"Strong transforms: {strong_transforms}")
    args.logger.info(f"Weak transforms: {weak_transforms}")
    args.logger.info(f"Test transforms: {test_transforms}")
    args.strong_transforms = strong_transforms
    args.weak_transforms = weak_transforms
    args.test_transforms = test_transforms

    # dataset
    train_labeled, train_unlabeled, testset = get_dataset(args)
    args.logger.info(f"Labeled train set: {train_labeled}")
    args.logger.info(f"Unlabeled train set: {train_unlabeled}")
    args.logger.info(f"Test set: {testset}")
    args.train_labeled = train_labeled
    args.train_unlabeled = train_unlabeled
    args.testset = testset

    # loader
    trainloader, trainloader_u, testloader = get_loaders(args)
    args.trainloader = trainloader
    args.trainloader_u = trainloader_u
    args.testloader = testloader

    # model
    model = get_model(args)
    args.logger.info(f"Model size: {model.get_param_size()}")
    args.model = model
    args.model.to(args.device)

    # optimizer & scheduler
    optimizer, scheduler = get_optim(args)
    args.logger.info(f"Optimizer: {optimizer}")
    args.logger.info(f"Scheduler: {scheduler}")
    args.optimizer = optimizer
    args.scheduler = scheduler

    # criterion
    criterion = get_criterion(args)
    args.logger.info(f"Criterion: {criterion}")
    args.criterion = criterion

    # train params
    start_epoch = 0
    args.best_val_acc = 0.0
    args.best_val_loss = 100.0

    # resume
    if args.cfg.train.resume:
        args, params = load_model(args)
        start_epoch, args.best_val_acc, args.best_val_loss = params
    
    # check eval
    if not args.cfg.train.skip_first_eval:
        args.current_epoch = -1
        args = eval(args)
    
    # main training loop
    for ep in range(start_epoch, args.cfg.train.num_epochs):
        args.current_epoch = ep
        # train
        args = train(args)
        # eval
        args = eval(args)
        # print
        args.logger.info("Epoch: {}\nTrain Acc: {:.4f}\tTrain Loss: {:.4f}\nVal Acc: {:.4f}\tVal Loss: {:.4f}".format(
            args.current_epoch, args.train_acc, args.train_loss, args.val_acc, args.val_loss
        ))
        # save
        save_model(args)
        # log to tensorboard
        args.writer.add_scalar("Avg train acc", args.train_acc)
        args.writer.add_scalar("Avg train loss", args.train_loss)
        args.writer.add_scalar("Avg val acc", args.val_acc)
        args.writer.add_scalar("Avg val loss", args.val_loss)


if __name__ == "__main__":
    # args
    args = Namespace()
    cfg = get_args()
    if cfg.transform.strong.RA and cfg.transform.strong.CTA:
        raise Exception("RA and CTA is not supported together")
    args.cfg = cfg
    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # main
    main(args)
