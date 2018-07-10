from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data

from data_loaders.cifar10_data_loader import CIFAR10DataLoader
from data_loaders.kth_data_loader import KTHDataLoader
from graph.ce_loss import Loss as Loss_ce
from graph.mse_loss import Loss as Loss_mse
from graph.ce_model import VAE as VAE_ce
from graph.mse_model import VAE as VAE_mse
from train.ce_trainer import Trainer as Trainer_ce
from train.mse_trainer import Trainer as Trainer_mse
from utils.utils import *
from utils.weight_initializer import Initializer
import pdb


def main():
    # Parse the JSON arguments
    args = parse_args()

    # Create the experiment directories
    args.summary_dir, args.checkpoint_dir = create_experiment_dirs(
        args.experiment_dir)

    if args.loss == 'ce':
        model = VAE_ce(args.input_shape.channels)
    else:
        model = VAE_mse(args.input_shape.n_channels)

    # to apply xavier_uniform:
    Initializer.initialize(model=model, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))

    if args.loss == 'ce':
        loss = Loss_ce()
    else:
        loss = Loss_mse()

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()
        loss.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    if args.dataset == 'CIFAR10':
        data = CIFAR10DataLoader(args)
    elif args.dataset == 'KTH':
        data = KTHDataLoader(args)
    print("Data loaded successfully\n")

    if args.loss == 'ce':
        trainer = Trainer_ce(model, loss, data.train_loader, data.test_loader, args)
    else:
        trainer = Trainer_mse(model, loss, data.train_loader, data.test_loader, args)

    if args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            print("Training had been Interrupted\n")

    if args.to_test:
        print("Testing on training data...")
        trainer.test_on_trainings_set()
        print("Testing Finished\n")


if __name__ == "__main__":
    main()
