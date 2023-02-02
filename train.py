import torch
import torchvision
import os
from argparse import Namespace
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import get_args
from utils import get_logger, get_transforms, get_dataset, get_loaders, get_model, get_optim, get_criterion, save_model, load_model


def train(args):
    args.model.train()
    running_acc = 0.0
    running_loss = 0.0
    num_samples = 0
    args.train_acc = 0.0
    args.train_loss = 0.0
    # choose longer loader to be main loader, to guarantee each data is seen at least once
    if len(args.trainloader_u) >= len(args.trainloader):
        unlabeled_main = True
        loader_gen = iter(args.trainloader)
    else:
        unlabeled_main = False
        loader_gen = iter(args.trainloader_u)
    pbar = tqdm(args.trainloader_u if unlabeled_main else args.trainloader)
    for i, (inputs, labels) in enumerate(pbar):
        # get labeled & unlabeled batch
        try:
            inputs_gen, labels_gen = next(loader_gen)  # labled data if unlabled_main is True, else unlabeled data
        except StopIteration:
            loader_gen = iter(args.trainloader if unlabeled_main else args.trainloader_u)
            inputs_gen, labels_gen = loader_gen.next()
        if unlabeled_main:
            images_u = inputs
            inputs_l, labels_l = inputs_gen.to(args.device), labels_gen.to(args.device)
        else:
            images_u = inputs_gen
            inputs_l, labels_l = inputs.to(args.device), labels.to(args.device)
        num_samples += inputs_l.shape[0]
        # forward labeled data
        args.optimizer.zero_grad()
        outputs = args.model(inputs_l).softmax(1)
        # compute loss for labeled data
        loss_l = args.criterion(outputs, torch.nn.functional.one_hot(labels_l, 10).float())
        running_acc = (outputs.argmax(1) == labels_l).sum().item()
        args.train_acc += running_acc
        num_samples += inputs_l.shape[0]
        # apply weak transform to unlabeled input data for acquiring pseudo label
        with torch.no_grad():
            tensors_u = []
            for img in images_u:
                tensors_u.append(args.weak_transforms(img))
            tensors_u = torch.stack(tensors_u, dim=0).to(args.device)
            outputs = args.model(tensors_u).softmax(1)
            max_values, max_index = outputs.max(1)
            pseudo_mask = max_values > args.cfg.train.tau
            pseudo_labels = max_index[pseudo_mask]
        # construct, forward, and compute loss for unlabled data if pseudo label exists
        if pseudo_labels.shape[0] > 0:
            inputs_u = []
            valid_index = pseudo_mask.nonzero()[:, 0].tolist()
            for vi in valid_index:
                inputs_u.append(args.strong_transforms(images_u[vi]))
                # img_grid = torchvision.utils.make_grid(inputs_u[-1])
                # args.writer.add_image('cta-transformed image', img_grid, i)
            inputs_u = torch.stack(inputs_u, dim=0).to(args.device)
            outputs = args.model(inputs_u).softmax(1)
            if args.cfg.transform.strong.CTA is True:
                args.strong_transforms.transforms[2].update(outputs, torch.nn.functional.one_hot(pseudo_labels, 10).float())
            loss_u = args.criterion(outputs, torch.nn.functional.one_hot(pseudo_labels, 10).float())
            loss_total = loss_l + loss_u * args.cfg.train.unsup_weight
        else:
            loss_total = loss_l
        # backward
        running_loss = loss_total.item()
        args.train_loss += running_loss
        loss_total.backward()
        args.optimizer.step()
        # print
        pbar.set_description("Running Acc:{:.4f}, Running Loss:{:.4f}, Avg Acc:{:.4f}, Avg Loss:{:.4f}".format(
            running_acc / inputs_l.shape[0], running_loss, args.train_acc / num_samples, args.train_loss / (i + 1)))
    # average train acc & loss
    args.train_acc = args.train_acc / num_samples
    args.train_loss = args.train_loss / (i + 1)  # computing accurate avg loss is difficult due to the varying batch size, use batch index i instead
    return args


def eval(args):
    args.model.eval()
    running_acc = 0.0
    running_loss = 0.0
    num_samples = 0
    args.val_acc = 0.0
    args.val_loss = 0.0
    args.cfg.param.checkpoint_name = "model_last.pth"

    pbar = tqdm(args.testloader)
    for i, (inputs, labels) in enumerate(pbar):
        num_samples += inputs.shape[0]
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        # forward
        outputs = args.model(inputs).softmax(1)
        # compute acc & loss
        running_acc = (outputs.argmax(1) == labels).sum().item()
        args.val_acc += running_acc
        loss = args.criterion(outputs, torch.nn.functional.one_hot(labels, 10).float())
        running_loss = loss.item()
        args.val_loss += running_loss * inputs.shape[0]
        # set tqdm description
        pbar.set_description("Running Acc:{:.4f}, Running Loss:{:.4f}, Avg Acc:{:.4f}, Avg Loss:{:.4f}".format(
            running_acc / inputs.shape[0], running_loss, args.val_acc / num_samples, args.val_loss / num_samples))
    # average val acc & loss
    args.val_acc = args.val_acc / num_samples
    args.val_loss = running_loss / num_samples
    # update best val acc
    if args.val_acc > args.best_val_acc:
        args.best_val_acc = args.val_acc
        args.cfg.param.checkpoint_name = "model_best_acc.pth"
    # update best val loss
    if args.val_loss < args.best_val_loss:
        args.best_val_loss = args.val_loss
        args.cfg.param.checkpoint_name = "model_best.pth"
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
        if params is not None:
            start_epoch, args.best_val_acc, args.best_val_loss = params

    # check eval
    if not args.cfg.train.skip_first_eval:
        args.current_epoch = -1
        args = eval(args)

    # main training loop
    if args.dataparallel:
        args.model = torch.nn.DataParallel(args.model)
    for ep in range(start_epoch, args.cfg.train.num_epochs):
        args.current_epoch = ep
        args.logger.info("Epoch: {}\tLR: {}".format(ep, args.optimizer.state_dict()['param_groups'][0]['lr']))
        # train
        args = train(args)
        # eval
        args = eval(args)
        # scheduler
        args.scheduler.step()
        # print
        args.logger.info("Train Acc: {:.4f}\tTrain Loss: {:.4f}\nVal Acc: {:.4f}\tVal Loss: {:.4f}".format(
            args.train_acc, args.train_loss, args.val_acc, args.val_loss
        ))
        # save
        save_model(args)
        args.cfg.param.checkpoint_name = "model_last.pth"
        save_model(args)
        # log to tensorboard
        args.writer.add_scalar("Avg train acc", args.train_acc, ep)
        args.writer.add_scalar("Avg train loss", args.train_loss, ep)
        args.writer.add_scalar("Avg val acc", args.val_acc, ep)
        args.writer.add_scalar("Avg val loss", args.val_loss, ep)
        args.writer.add_scalar("lr", args.scheduler.get_last_lr()[0], ep)
    args.logger.info("Best val acc: {:.4f} (epoch {})\tBest val loss: {:.4f} (epoch {})".format(args.best_val_acc, args.best_acc_ep, args.best_val_loss, args.best_loss_ep))


if __name__ == "__main__":
    # args
    args = Namespace()
    cfg = get_args()
    if cfg.transform.strong.RA and cfg.transform.strong.CTA:
        raise Exception("RA and CTA is not supported together")
    args.cfg = cfg
    os.makedirs(os.path.join(cfg.param.checkpoint_dir, cfg.param.log_name), exist_ok=True)
    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 2:
        args.dataparallel = True
    else:
        args.dataparallel = False
    # main
    main(args)
