
import argparse
import glob
import torch
import shutil
import rdkit
from torch import nn
# import args
from lightning.pytorch.utilities import rank_zero_warn, rank_zero_only
from lightning.pytorch.utilities import rank_zero_only
from molformer.model.args import get_parser as ARGS
import os
import numpy as np
import random
import getpass
from datasets import load_dataset, concatenate_datasets
from pubchem_encoder import Encoder
import lightning.pytorch as pl
# from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from torch.utils.data import DataLoader
import subprocess
import wandb
# from pytorch_lightning import LightningModule
from molformer.model.base_bert import LightningModule

from molformer.model.base_bert import fix_infiniband, remove_tree
from molformer.model.base_bert import MoleculeModule, LightningModule
from attributedict.collections import AttributeDict
from molformer.utils import get_argparse_defaults


def main(name):    
    fix_infiniband()
    config = AttributeDict(get_argparse_defaults(ARGS()))
    config.num_nodes = 1
    config.n_batch = 3200
    config.n_head = 12
    config.n_layer = 12
    config.n_embd = 768
    config.max_len = 202
    config.d_dropout = 0.2
    config.lr_start = 3e-5
    config.lr_multiplier = 8
    config.n_workers = 8
    config.max_epochs = 4
    config.gpu = 1
    config.num_nodes = 1
    config.accelerator = 'ddp' 
    config.num_feats = 32
    config.root_dir = './molformer_LARGE'
    config.checkpoint_every = 1000
    config.train_load = 'both'
    config.mode = 'cls'
    config.attention_type = 'full'
    # config.restart_path = "../../../molformer_XL/checkpoints/checkpoint_2_13000.ckpt"
    # config.restart_path = "../molformer_refactor/data/checkpoints/N-Step-Checkpoint_3_30000.ckpt"
    config.restart_path = ""
    if config.num_nodes > 1:
        # print("Using " + str(config.num_nodes) + " Nodes----------------------------------------------------------------------")
        LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ') # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
        HOST_LIST = LSB_MCPU_HOSTS[::2] # Strips the cores per node items in the list
        os.environ["MASTER_ADDR"] = HOST_LIST[0] # Sets the MasterNode to thefirst node on the list of hosts
        os.environ["MASTER_PORT"] = "54966"
        os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"])) #Uses the list index for node rank, master node rank must be 0
        #os.environ["NCCL_SOCKET_IFNAME"] = 'ib,bond'  # avoids using docker of loopback interface
        os.environ["NCCL_DEBUG"] = "INFO" #sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
        #os.environ["NCCL_IB_CUDA_SUPPORT"] = '1' #Force use of infiniband
        #os.environ["NCCL_TOPO_DUMP_FILE"] = 'NCCL_TOP.%h.xml'
        #os.environ["NCCL_DEBUG_FILE"] = 'NCCL_DEBUG.%h.%p.txt'
        print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
        print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
        print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
        print(os.environ["HOSTNAME"] + " NCCL_SOCKET_IFNAME: " + os.environ["NCCL_SOCKET_IFNAME"])
        print(os.environ["HOSTNAME"] + " NCCL_DEBUG: " + os.environ["NCCL_DEBUG"])
        print(os.environ["HOSTNAME"] + " NCCL_IB_CUDA_SUPPORT: " + os.environ["NCCL_IB_CUDA_SUPPORT"])
        print("Using " + str(config.num_nodes) + " Nodes---------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    else:
        print("Using " + str(config.num_nodes) + " Node----------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")


    checkpoint_dir = os.path.join(config.root_dir, 'checkpoints')
    train_config = {'batch_size': config.n_batch, 'num_workers':config.n_workers, 'pin_memory':True}
    # this should allow us to save a model for every x iterations and it should overwrite
    
    from base_bert import ExponentialScheduleCallback
    
    checkpoint_callback = ExponentialScheduleCallback(save_top_k = -1, 
                                                      dirpath=checkpoint_dir, 
                                                      filename='molformer', 
                                                      every_n_train_steps = config.checkpoint_every,
                                                      verbose=True, 
                                                      limit_step=1000)
    
    train_loader = MoleculeModule(config.max_len, config.train_load, train_config)
    train_loader.setup()#config.debug)
    cachefiles = train_loader.get_cache()
    model = LightningModule(config, train_loader.get_vocab())
    
    trainer = pl.Trainer(default_root_dir=config.root_dir,
                max_epochs=config.max_epochs,
                accelerator=config.device,
                # strategy="ddp",
                num_nodes=config.num_nodes,
                # gpus=config.gpus,
                callbacks=[checkpoint_callback],
                enable_checkpointing=True,
                accumulate_grad_batches=config.grad_acc,
                log_every_n_steps=config.checkpoint_every,
                # val_check_interval=10, 
                devices=1)
    
                # weights_summary='full')
    # try: 
    
    trainer.train(model, train_loader)#, ckpt_path = config.restart_path)

    trainer.fit(model, train_loader)
    
    # except Exception as exp: 
    #     # exit()
    #     rank_zero_warn('We have caught an error, trying to shut down gracefully')
    #     remove_tree(cachefiles)

    if config.debug is True:
        pass
    else:
        # exit()
        rank_zero_warn('Debug mode not found eraseing cache')
        remove_tree(cachefiles)


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('high')
    wandb.login()
    training_points = "find_test_loss"
    wandb.init(project=f'molformer_full_attention')##, id = 'test1')
            #    id='training_937MB_pubchem')
    main(name = "XL")
    wandb.finish()
