#!/usr/bin/env python
# coding: utf-8

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import glob
import re
import dataloader_FiLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchinfo import summary
import argparse
from natsort import natsorted
import librosa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from module_FiLM import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=int(16000*2), help="cut length, default is 2 seconds in denoise and dereverberation")
parser.add_argument("--save_model_dir", type=str, default='/home/iasp_guest1/tzx/code/BSRNN/allf0316', help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.5, 0.5, 1], help="weights of RI components, magnitude, and Metric Disc")

args, _ = parser.parse_known_args()
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 512
        self.hop = 128
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.window = torch.hann_window(self.n_fft).cuda()
        
        self.model = BSRNN(num_channel=64, num_layer=5, clap_dim=768).cuda()
        self.discriminator = Discriminator(ndf=16).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=args.init_lr)
        
    def get_latest_ckpt(self):
        if not os.path.exists(args.save_model_dir):
            return -1, None, None

        gene_ckpts = glob.glob(os.path.join(args.save_model_dir, 'gene_epoch_*'))
        if not gene_ckpts:
            return -1, None, None

        latest_epoch = -1
        latest_gene_ckpt = None
        
        for ckpt in gene_ckpts:
            match = re.search(r'gene_epoch_(\d+)', ckpt)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_gene_ckpt = ckpt

        latest_disc_ckpt = os.path.join(args.save_model_dir, f'disc_epoch_{latest_epoch}')
        if not os.path.exists(latest_disc_ckpt):
            latest_disc_ckpt = None

        return latest_epoch, latest_gene_ckpt, latest_disc_ckpt

    def train_step(self, batch, use_disc):
        clean = batch[0].cuda(non_blocking=True)
        noisy = batch[1].cuda(non_blocking=True)
        f_a = batch[2].cuda(non_blocking=True)
        f_t = batch[3].cuda(non_blocking=True)
        one_labels = torch.ones(clean.size(0)).cuda(non_blocking=True)
    
        self.optimizer.zero_grad()
        
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=self.window, onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=self.window, onesided=True, return_complex=True)
                
        est_spec = self.model(noisy_spec, f_a=f_a, f_t=f_t)
        
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)
        
        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * gen_loss_GAN

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()
        
        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=self.window, onesided=True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)

        if pesq_score is not None and pesq_score_n is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)            
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels.float()) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)

            discrim_loss_metric.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])
                
        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch, use_disc):
        clean = batch[0].cuda(non_blocking=True)
        noisy = batch[1].cuda(non_blocking=True)
        f_a = batch[2].cuda(non_blocking=True)
        f_t = batch[3].cuda(non_blocking=True)
        one_labels = torch.ones(clean.size(0)).cuda(non_blocking=True)

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=self.window, onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=self.window, onesided=True, return_complex=True)
        
        # 传入先验特征
        est_spec = self.model(noisy_spec, f_a=f_a, f_t=f_t)
        
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** (0.3)
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** (0.3)
        noisy_mag = (torch.abs(noisy_spec).unsqueeze(1) + 1e-10) ** (0.3)

        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        if use_disc is False:
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag
        else:
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * gen_loss_GAN

        est_audio = torch.istft(est_spec, self.n_fft, self.hop, window=self.window, onesided=True)

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy())
        noisy_audio_list = list(noisy.cpu().numpy())
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_n = batch_pesq(est_audio_list, noisy_audio_list)
        
        if pesq_score is not None and pesq_score_n is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            predict_min_metric = self.discriminator(est_mag.detach(), noisy_mag)            
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss(predict_min_metric.flatten(), pesq_score_n)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self, use_disc):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(tqdm(self.test_ds)):
            step = idx + 1
            loss, disc_loss = self.test_step(batch, use_disc)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = 'Generator loss: {:.4f}, Discriminator loss: {:.4f}'
        logging.info(template.format(gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        def lr_lambda(epoch):
            if epoch < args.decay_epoch:
                return 1.0
            else:
                return 0.98 ** (epoch - args.decay_epoch)

        scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_disc, lr_lambda)

        start_epoch = 0
        latest_epoch, latest_gene_ckpt, latest_disc_ckpt = self.get_latest_ckpt()

        if latest_epoch >= 0:
            logging.info(f"🔍 Found checkpoint from epoch {latest_epoch}. Resuming training...")
            
            checkpoint_G = torch.load(latest_gene_ckpt)
            if isinstance(checkpoint_G, dict) and 'model_state_dict' in checkpoint_G:
                self.model.load_state_dict(checkpoint_G['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint_G['optimizer_state_dict'])
                scheduler_G.load_state_dict(checkpoint_G['scheduler_state_dict'])
            else:
                self.model.load_state_dict(checkpoint_G)
                scheduler_G.last_epoch = latest_epoch

            if latest_disc_ckpt:
                checkpoint_D = torch.load(latest_disc_ckpt)
                if isinstance(checkpoint_D, dict) and 'model_state_dict' in checkpoint_D:
                    self.discriminator.load_state_dict(checkpoint_D['model_state_dict'])
                    self.optimizer_disc.load_state_dict(checkpoint_D['optimizer_state_dict'])
                    scheduler_D.load_state_dict(checkpoint_D['scheduler_state_dict'])
                else:
                    self.discriminator.load_state_dict(checkpoint_D)
                    scheduler_D.last_epoch = latest_epoch

            start_epoch = latest_epoch + 1
        else:
            logging.info("🌟 No checkpoint found. Starting training from scratch...")

        for epoch in range(start_epoch, args.epochs):
            self.model.train()
            self.discriminator.train()

            loss_total = 0
            loss_gan = 0
            
            if epoch >= (args.epochs / 2):
                use_disc = True
            else:
                use_disc = False
            
            for idx, batch in enumerate(tqdm(self.train_ds)):
                step = idx + 1
                loss, disc_loss = self.train_step(batch, use_disc)
                template = 'Epoch {}, Step {}, loss: {:.4f}, disc_loss: {:.4f}'
                
                loss_total += loss
                loss_gan += disc_loss
                
                if (step % args.log_interval) == 0:
                    logging.info(template.format(epoch, step, loss_total/step, loss_gan/step))

            gen_loss = self.test(use_disc)
            
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)

            path = os.path.join(args.save_model_dir, 'gene_epoch_' + str(epoch) + '_' + str(gen_loss)[:5])
            path_d = os.path.join(args.save_model_dir, 'disc_epoch_' + str(epoch))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_G.state_dict(),
                'gen_loss': gen_loss
            }, path)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer_disc.state_dict(),
                'scheduler_state_dict': scheduler_D.state_dict(),
            }, path_d)

            scheduler_G.step()
            scheduler_D.step()

def main():
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    train_ds, test_ds = F2dataloader111.load_data(args.batch_size, 2, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()

if __name__ == '__main__':
    main()
