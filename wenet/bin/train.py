# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
from inspect import getargs
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import (load_checkpoint, save_checkpoint,
                                    load_trained_modules)
from wenet.utils.executor import Executor
from wenet.utils.file_utils import read_lists, read_symbol_table, read_non_lang_symbols
from wenet.utils.scheduler import WarmupLR, DistillWeightLR
from wenet.utils.config import override_config


# NOTE(xcsong): Enable optimization for convolution.
#               ref: https://zhuanlan.zhihu.com/p/73711222
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--hdfs_dir',
                        default=None,
                        help='save model to hdfs dir')
    parser.add_argument('--hdfs_username',
                        default='ziyu.wang',
                        help='your hdfs user name')
    parser.add_argument('--local_run',
                        action='store_true',
                        default=False,
                        help='running on local or aidi cluster')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--device-ids',
                        type=str,
                        required=False,
                        default=None,
                        help="GPU device ids like 0,1,2,3")
    parser.add_argument(
        '--dist_url',
        type=str,
        required=False,
        default='auto',
        help='dist url for init process, such as tcp://localhost:8000')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument("--enc_init_mods",
                        default="encoder.",
                        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
                        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument('--train_utt',
                        type=str,
                        help='')
    parser.add_argument('--finetune',
                        action='store_true',
                        help='')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.local_run:
        client = None
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        import mpi4py.MPI as MPI
        from wenet.dataset.hdfs_io import HdfsCli
        client = HdfsCli(args.hdfs_username)
        comm = MPI.COMM_WORLD
        local_rank = comm.Get_rank()
        world_size = comm.Get_size()

    device_ids = args.device_ids.split(',')
    num_devices = len(device_ids)
    current_device = local_rank % num_devices
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids[current_device]
    args.gpu = int(device_ids[current_device])

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    distributed = world_size > 1
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        try:
            if args.local_run:
                # torchrun: no need for you to pass RANK manually
                dist.init_process_group(backend=args.dist_backend)
            else:
                dist.init_process_group(backend=args.dist_backend,
                                        init_method=args.dist_url,
                                        world_size=world_size,
                                        rank=local_rank)
        except Exception as e:
            logging.info("Process group URL: {}".format(args.dist_url))
            raise e

    symbol_table = read_symbol_table(args.symbol_table)

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    # NOTE(xcsong): DDP stucks when using `dynamic batch` + `shuffle=True`.
    #   see https://github.com/wenet-e2e/wenet/issues/842#issuecomment-999219266
    cv_conf['batch_conf']['batch_type'] = 'static'
    cv_conf['shuffle'] = False
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    train_dataset = Dataset(args.data_type, args.train_data, symbol_table,
                            train_conf, args.bpe_model, non_lang_syms, True)
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         symbol_table,
                         cv_conf,
                         args.bpe_model,
                         non_lang_syms,
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    train_utt_list = read_lists(args.train_utt)
    batch_size = configs['dataset_conf']['batch_conf']['batch_size']
    loadersize = len(train_utt_list) // world_size // batch_size
    logging.info("train data: {}".format(loadersize))

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    vocab_size = len(symbol_table)

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True
    if local_rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)
        if args.hdfs_dir is not None:
            client.upload_file(args.hdfs_dir, saved_config_path)

    # Init asr model from configs
    model = init_asr_model(configs)
    # print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))
    num_params_encoder = 0
    num_params_decoder = 0
    for name, param in model.named_parameters():
        if "encoder." in name:
            num_params_encoder += param.numel()
        elif "decoder." in name:
            num_params_decoder += param.numel()
    print('the number of encoder params: {}'.format(num_params_encoder))
    print('the number of decoder params: {}'.format(num_params_decoder))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements

    # There's bug in FSMNEncoder and fast u2++
    # if local_rank == 0:
    #     script_model = torch.jit.script(model)
    #     script_model.save(os.path.join(args.model_dir, 'init.zip'))
    #     if args.hdfs_dir is not None:
    #         client.upload_file(args.hdfs_dir,
    #                            os.path.join(args.model_dir, 'init.zip'))

    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif args.enc_init is not None:
        logging.info('load pretrained encoders: {}'.format(args.enc_init))
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    if args.finetune:
        # add freeze params
        for n, p in model.named_parameters():
            if 'l_ctc' in n or 'encoder.l_encoders' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
            print('{} {}'.format(n, p.requires_grad))

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))
    ori_model = model
    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
        if args.fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default as comm_hooks,
            )
            model.register_comm_hook(
                state=None, hook=comm_hooks.fp16_compress_hook
            )
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])

    # add distill_weight scheduler
    configs['weight_update']['epoch_iter'] = loadersize
    distill_weight_scheduler = DistillWeightLR(
        model=ori_model, **configs['weight_update'])
    final_epoch = None
    args.rank = local_rank
    configs['rank'] = local_rank
    configs['is_distributed'] = distributed
    configs['use_amp'] = args.use_amp
    if start_epoch == 0 and local_rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)
        if args.hdfs_dir is not None:
            client.upload_file(args.hdfs_dir, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)

        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, distill_weight_scheduler, train_data_loader,
                       cv_data_loader, device, writer, configs, scaler, client,
                       model_dir, args.hdfs_dir)
        total_loss, total_ctc_loss, total_att_acc, num_seen_utts = executor.cv(
            model, cv_data_loader, device, configs)

        cv_loss = total_loss / num_seen_utts
        ctc_loss = total_ctc_loss / num_seen_utts
        att_acc = total_att_acc / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        if local_rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_yaml_path = os.path.join(model_dir, '{}.yaml'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'ctc_loss': ctc_loss,
                    'att_acc': att_acc,
                    'step': executor.step
                })
            writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            if args.hdfs_dir is not None:
                client.upload_file(args.hdfs_dir, save_model_path)
                client.upload_file(args.hdfs_dir, save_yaml_path)
        final_epoch = epoch
        # empty cache on end of epochs
        torch.cuda.empty_cache()

    if final_epoch is not None and local_rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        if args.hdfs_dir is not None:
            client.upload_file(args.hdfs_dir, final_model_path)
        writer.close()


if __name__ == '__main__':
    main()
