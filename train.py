"""Train the model"""

import argparse
import datetime
import os
import itertools
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1 ,2'
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.losses import compute_losses

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/RealHomo/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default='',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")
parser.add_argument('-ow', '--only_weights', action='store_true', default=True,
                    help='Only use weights to load or load all train status.')
parser.add_argument('--seed', type=int, default=230, help='random seed')


def train(model, manager):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # loss status and val/test status initial
    manager.reset_loss_status()
    # set model to training mode
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)

    model.train()

    with tqdm(total=len(manager.dataloaders['train']), ncols=200) as t:
        for i, data_batch in enumerate(manager.dataloaders['train']):
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)
            print_str = manager.print_train_info()

            output = model(data_batch)

            loss = {}
            loss.update(compute_losses(data_batch, output, manager.params))
            manager.update_loss_status(loss=loss, split="train")
            manager.optimizer.zero_grad()
            loss['total'].backward()
            manager.optimizer.step()

            # save loss in trainlog
            manager.update_step()
            if i % manager.params.eval_freq == 0 and i != 0:
                print('\n')
                val_metrics = evaluate(model, manager)
                avg = val_metrics['MSE_avg']
                manager.cur_val_score = avg
                manager.check_best_save_last_checkpoints(latest_freq=1)

            t.set_description(desc=print_str)
            t.update()

    manager.scheduler.step()
    manager.update_epoch()


def train_and_evaluate(model, manager):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
    """

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()

    for epoch in range(manager.params.num_epochs):
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)
        evaluate(model, manager)
        manager.check_best_save_last_checkpoints(latest_freq=1)


if __name__ == '__main__':
    # Load the parameters from json file
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Update args into params
    params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    if params.cuda:
        torch.cuda.manual_seed(args.seed)

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Loading the datasets from {}".format(params.data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)
    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

    # initial status for checkpoint manager
    manager = Manager(model=model, optimizer=optimizer,
                      scheduler=scheduler, params=params, dataloaders=dataloaders,
                      writer=None, logger=logger)

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, manager)
