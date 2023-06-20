"""Evaluates the model"""

import argparse
import logging
import os
import torch.optim as optim
import torch
from tqdm import tqdm
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from loss.losses import compute_eval_results
from common.manager import Manager

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/RealHomo/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default='experiments/RealHomo/EM2/EM2_0.3445.pth',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")
parser.add_argument('-ow', '--only_weights', action='store_true', default=True,
                    help='Only use weights to load or load all train status.')


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode
    manager.logger.info("eval begin!")

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    torch.cuda.empty_cache()
    model.eval()
    k = 0
    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("val")

            for data_batch in manager.dataloaders["val"]:
                # move to GPU if available

                video_name = data_batch["video_names"]

                data_batch = utils.tensor_gpu(data_batch)
                output = model(data_batch)

                # compute all loss on this batch
                eval_results = compute_eval_results(data_batch, output)
                err_avg = eval_results["errors_m"]

                for j in range(len(err_avg)):
                    k += 1
                    if video_name[j] in RE:
                        MSE_RE.append(err_avg[j])
                    elif video_name[j] in LT:
                        MSE_LT.append(err_avg[j])
                    elif video_name[j] in LL:
                        MSE_LL.append(err_avg[j])
                    elif video_name[j] in SF:
                        MSE_SF.append(err_avg[j])
                    elif video_name[j] in LF:
                        MSE_LF.append(err_avg[j])

        MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
        MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
        MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
        MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
        MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
        MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5

        Metric = {"MSE_RE_avg": MSE_RE_avg, "MSE_LT_avg": MSE_LT_avg, "MSE_LL_avg": MSE_LL_avg,
                  "MSE_SF_avg": MSE_SF_avg, "MSE_LF_avg": MSE_LF_avg, "AVG": MSE_avg}
        manager.update_metric_status(metrics=Metric, split="val")

        manager.logger.info(
            "Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                manager.epoch_val,
                MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

        manager.print_metrics("val", title="val", color="green")

        manager.epoch_val += 1

        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        model.train()
        val_metrics = {'MSE_avg': MSE_avg}
        return val_metrics


def test(model, manager):
    """Test the model with loading checkpoints.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    torch.cuda.empty_cache()
    model.eval()
    k = 0
    flag = 0

    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["test"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("test")
            with tqdm(total=len(manager.dataloaders['test']), ncols=100) as t:
                for data_batch in manager.dataloaders["test"]:

                    video_name = data_batch["video_names"]
                    data_batch = utils.tensor_gpu(data_batch)
                    output_batch = model(data_batch)

                    flag += 1
                    t.update()
                    eval_results = compute_eval_results(data_batch, output_batch)
                    err_avg = eval_results["errors_m"]
                    for j in range(len(err_avg)):
                        k += 1
                        if video_name[j] in RE:
                            MSE_RE.append(err_avg[j])
                        elif video_name[j] in LT:
                            MSE_LT.append(err_avg[j])
                        elif video_name[j] in LL:
                            MSE_LL.append(err_avg[j])
                        elif video_name[j] in SF:
                            MSE_SF.append(err_avg[j])
                        elif video_name[j] in LF:
                            MSE_LF.append(err_avg[j])

            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5

            Metric = {"MSE_RE_avg": MSE_RE_avg, "MSE_LT_avg": MSE_LT_avg, "MSE_LL_avg": MSE_LL_avg,
                      "MSE_SF_avg": MSE_SF_avg, "MSE_LF_avg": MSE_LF_avg, "AVG": MSE_avg}
            manager.update_metric_status(metrics=Metric, split="test")

            # update data to tensorboard
            manager.logger.info(
                "Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                    manager.epoch_val,
                    MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

            manager.print_metrics("test", title="test", color="red")


if __name__ == '__main__':

    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
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

    # Initial status for checkpoint manager

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    test(model, manager)
