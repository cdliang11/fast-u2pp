# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
import time
import os
import sys
import torch
from torch.nn.utils import clip_grad_norm_

from wenet.utils.checkpoint import save_checkpoint

if sys.version_info.minor >= 7:
    from contextlib import nullcontext
else:
    # if your python version < 3.7 use the below one
    from contextlib import suppress as nullcontext


class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, distill_weight_scheduler, data_loader, cv_data_loader,
              device, writer, args, scaler, client, model_dir, hdfs_dir):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        torch.cuda.synchronize()
        start = time.time()
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                # with torch.cuda.device(device):
                #     torch.cuda.empty_cache()
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    # torch.cuda.synchronize()
                    # t1 = time.time()
                    # logging.info('rank %d : feat to cuda %f' % (rank, t1-t0))
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss, loss_att, loss_ctc, acc_att, loss_distill, distill_weight, loss_distill_list = model(
                            feats, feats_lengths, target, target_lengths)
                        loss = loss / accum_grad

                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar('train_loss', loss.item(), self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in the
                        # iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient has
                        # inf/nan values, scaler.step will skip optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    distill_weight_scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    if loss_att is not None:
                        if len(loss_att) == 3:
                            log_str += 'loss_att (s_l: {:.6f} s_h: {:.6f} ns_h: {:.6f}) '.format(
                                loss_att[0].item(), loss_att[1].item(), loss_att[2].item())
                        elif len(loss_att) == 2:
                            log_str += 'loss_att (s_l: {:.6f} s_h: {:.6f}) '.format(
                                loss_att[0].item(), loss_att[1].item())
                    if loss_ctc is not None:
                        if len(loss_ctc) == 3:
                            log_str += 'loss_ctc (s_l: {:.6f} s_h: {:.6f} ns_h: {:.6f}) '.format(
                                loss_ctc[0].item(), loss_ctc[1].item(), loss_ctc[2].item())
                        elif len(loss_ctc) == 2:
                            log_str += 'loss_ctc (s_l: {:.6f} s_h: {:.6f}) '.format(
                                loss_ctc[0].item(), loss_ctc[1].item())
                    log_str += 'loss_distill {:.6f} '.format(loss_distill)
                    log_str += '('
                    for _loss_distill in loss_distill_list:
                        log_str += '{:.6f} '.format(_loss_distill)
                    log_str += ') '
                    log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                    log_str += ' att_acc (s_l: {:.3f} s_h: {:.3f} ns_h: {:.3f})'.format(
                        acc_att[0], acc_att[1], acc_att[2])
                    log_str += ' distill_weight: {:.6f}'.format(distill_weight)

                    torch.cuda.synchronize()
                    end = time.time()
                    speed = 1.0 * log_interval / (end - start)
                    log_str += ' speed {:.2f} batch/s'.format(speed)
                    start = end
                    logging.debug(log_str)

                if (batch_idx + 1) % 20000 == 0:
                    total_loss, total_ctc_loss, total_att_acc, num_seen_utts = self.cv(
                        model, cv_data_loader, device, args)
                    cv_loss = total_loss / num_seen_utts
                    cv_ctc_loss = total_ctc_loss / num_seen_utts
                    cv_att_acc = total_att_acc / num_seen_utts

                    if rank == 0:
                        # save model to hdfs every 20,000 steps
                        save_model_path = os.path.join(
                            model_dir, '{}_{}.pt'.format(epoch, batch_idx))
                        save_yaml_path = os.path.join(
                            model_dir, '{}_{}.yaml'.format(epoch, batch_idx))
                        save_checkpoint(
                            model, save_model_path, {
                                'epoch': epoch,
                                'lr': lr,
                                'cv_loss': cv_loss,
                                'ctc_loss': cv_ctc_loss,
                                'att_acc': cv_att_acc,
                                'step': self.step
                            })
                        if hdfs_dir is not None:
                            client.upload_file(hdfs_dir, save_model_path)
                            client.upload_file(hdfs_dir, save_yaml_path)

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        total_loss_ctc = 0.0
        total_acc_att = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, loss_ctc, acc_att, loss_distill, _, loss_distill_list = model(
                    feats, feats_lengths, target, target_lengths)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                    total_loss_ctc += (loss_ctc[0].item() + loss_ctc[1].item() + loss_ctc[2].item()) * num_utts
                    total_acc_att += (acc_att[0] + acc_att[1] + acc_att[2]) / 2 * num_utts if acc_att is not None else 0.0
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att (s_l {:.6f} s_h {:.6f} ns_h {:.6f}) '.format(loss_att[0].item(), loss_att[1].item(), loss_att[2].item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc (s_l {:.6f} s_h {:.6f}) ns_h {:.6f}'.format(loss_ctc[0].item(), loss_ctc[1].item(), loss_ctc[2].item())
                    log_str += 'loss_distill {:.6f}'.format(loss_distill)
                    log_str += '('
                    for _loss_distill in loss_distill_list:
                        log_str += '{:.6f} '.format(_loss_distill)
                    log_str += ') '
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)

        return total_loss, total_loss_ctc, total_acc_att, num_seen_utts
