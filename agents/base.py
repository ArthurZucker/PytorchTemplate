"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import shutil
from os import path

import numpy as np
import torch
# visualisation tool
import wandb
from datasets import *
from datasets.BirdsDataloader import BirdsDataloader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from utils.agent_utils import get_loss, get_net, get_optimizer
# import your classes here
from utils.metrics import AverageMeter, cls_accuracy, compute_metrics
from utils.misc import print_cuda_statistics


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config, run):
        self.config = config
        self.model = get_net(config)
        print(self.model)
        run.watch(self.model)  # run is a wandb instance
        self.data_loader = globals()[self.config.dataloader](self.config)

        # define loss
        self.loss = get_loss(config.loss)

        # define optimizers for both generator and discriminator
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            print("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            print("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(
                self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + "/" + filename,
                            self.config.checkpoint_dir + "/" + f'{wandb.run.name}_model_best_{self.best_valid_acc.numpy():.2f}.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        max_epoch = self.config.max_epoch
        if self.config.test_mode:
            max_epoch = 2

        for epoch in range(self.current_epoch, min(max_epoch, self.config.max_epoch)):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.scheduler.step()
            if epoch % self.config.validate_every == 0:
                valid_acc = self.validate()
                is_best = valid_acc > self.best_valid_acc
                if is_best:
                    self.best_valid_acc = valid_acc
                self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch of training
        uses tqdm to load data in parallel? Nope thats not true
        :return:
        """
        if self.config.test_mode:
            self.data_loader.train_iterations = 10

        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch), leave=True)
        # Set the model to be in training mode
        self.model.train()
        epoch_loss = AverageMeter()
        correct = AverageMeter()
        for current_batch, (x, y) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(
                    non_blocking=self.config.async_loading)
            self.optimizer.zero_grad()
            pred = self.model(x)
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            cur_loss.backward()
            self.optimizer.step()
            epoch_loss.update(cur_loss.item())
            pred = pred.data.max(1, keepdim=True)[1]
            correct.update(
                pred.eq(y.data.view_as(pred)).cpu().sum()/y.shape[0])
            self.current_iteration += 1
            # logging in wand
            wandb.log({"epoch/loss": epoch_loss.val,
                       "epoch/accuracy": correct.val
                       })

            if self.config.test_mode and current_batch == 11:
                break

        tqdm_batch.close()
        print("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + "- Top1 Acc: " + str(correct.val))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        if self.config.test_mode:
            self.data_loader.valid_iterations = 5

        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Validation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        correct = AverageMeter()
        for current_batch, (x, y) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(
                    non_blocking=self.config.async_loading)
            pred = self.model(x)
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

            epoch_loss.update(cur_loss.item())
            pred = pred.data.max(1, keepdim=True)[1]
            correct.update(
                pred.eq(y.data.view_as(pred)).cpu().sum()/y.shape[0])
            dic = {}
            # dic = compute_metrics(output.cpu(), y.detach().cpu(),self.config.num_classes)
            dic.update({"epoch/validation_loss": epoch_loss.val,
                        "epoch/validation_accuracy": correct.val
                        })
            wandb.log(dic)

            if self.config.test_mode and current_batch == 5:
                break
        print("Validation results at epoch-" + str(self.current_epoch)
              + " | " + "loss: " + str(epoch_loss.avg)
              + "- Top1 Acc: " + str(correct.val))

        tqdm_batch.close()

        return correct.val

    def test(self):
        import PIL.Image as Image

        def pil_loader(path):
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB')

        # set the model in training mode
        self.model.eval()
        output_file = open(self.config.outfile, "w")
        output_file.write("Id,Category\n")
        for f in tqdm(os.listdir(self.config.test_dir)):
            if 'jpg' in f:
                data = self.data_loader.transform(
                    pil_loader(self.config.test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                if self.cuda:
                    data = data.cuda()
                output = self.model(data)
                pred = output.data.max(1, keepdim=True)[1]
                output_file.write("%s,%d\n" % (f[:-4], pred))
        output_file.close()
        print("Succesfully wrote " + self.config.outfile +
              ', you can upload this file to the kaggle competition website')

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
