from attributedict.collections import AttributeDict
from molformer.utils import get_argparse_defaults
from args import get_parser as ARGS
import torch
from bert_classification import PropertyPredictionDataModule, LightningModule
import lightning.pytorch as pl
import os
import numpy as np
import wandb
from molformer.tokenizer import MolTranBertTokenizer
from lightning.pytorch import seed_everything
import time
from bert_classification import ModelCheckpointAtEpochEnd, CheckpointEveryNSteps

def main():

    config = AttributeDict(get_argparse_defaults(ARGS()))
    config.device = 'cuda'
    config.batch_size = 128
    config.n_head = 12
    config.n_layer = 12
    config.n_embd = 768
    config.d_dropout = 0.1
    config.dropout = 0.1
    config.lr_start = 3e-5
    config.num_workers = 8
    config.max_epochs = 50
    config.num_feats = 32
    config.seed_path = '../../../../../molformer/data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
    config.dataset_name = 'bace'
    config.data_root = '../../../../data/bace'
    config.measure_name = 'RingCount'
    config.dims = [768, 768, 768, 1]
    config.checkpoints_folder = './checkpoints_bace__RingCount_more_val'
    config.num_classes = 2


    print("Using " + str(
        torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    pos_emb_type = 'rot'
    print('pos_emb_type is {}'.format(pos_emb_type))

    run_name_fields = [
        config.dataset_name,
        config.measure_name,
        pos_emb_type,
        config.fold,
        config.mode,
        "lr",
        config.lr_start,
        "batch",
        config.batch_size,
        "drop",
        config.dropout,
        config.dims,
    ]

    run_name = "_".join(map(str, run_name_fields))

    print("RUN", run_name)
    arguments = dict(config)
    datamodule = PropertyPredictionDataModule(arguments)
    config.dataset_names = "valid test".split()
    config.run_name = run_name

    checkpoints_folder = config.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder)
    config.checkpoint_root = checkpoint_root
    # config.run_id=np.random.randint(30000)
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "model")
    results_dir = os.path.join(checkpoint_root, "results")
    config.results_dir = results_dir
    config.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    # os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = MolTranBertTokenizer('../../data/bert_vocab.txt')
    seed_everything(config.seed)

    if config.seed_path == '':
        print("# training from scratch")
        model = LightningModule(config, tokenizer)
    else:
        print(f"# loaded pre-trained model from {config.seed_path}")
        model = LightningModule(config, tokenizer).load_from_checkpoint(config.seed_path, 
                                                                       strict=False,    
                                                                       config=config, 
                                                                       tokenizer=tokenizer, 
                                                                       vocab=len(tokenizer.vocab))

    last_checkpoint_file = config.seed_path 
    
    from molformer.model.base_bert import ExponentialScheduleCallback
    
    checkpoint_callback = ExponentialScheduleCallback(save_top_k = -1, 
                                                      dirpath=checkpoint_dir, 
                                                      filename='molformer', 
                                                      every_n_train_steps = config.checkpoint_every,
                                                      verbose=True, 
                                                      limit_step=50000)
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=100,
        default_root_dir=checkpoint_root,
        accelerator= config.device,
        devices=1,
        callbacks =  [checkpoint_callback],
        enable_checkpointing= True,
        num_sanity_val_steps=10
        )
    
    # trainer = pl.Trainer(default_root_dir=config.root_dir,
    #         max_epochs=config.max_epochs,
    #         accelerator="cuda",
    #         strategy="ddp",
    #         num_nodes=config.num_nodes,
    #         # gpus=config.gpus,
    #         callbacks=[ModelCheckpointAtEpochEnd(), CheckpointEveryNSteps(config.checkpoint_every)],
    #         enable_checkpointing=checkpoint_callback,
    #         accumulate_grad_batches=config.grad_acc,
    #         num_sanity_val_steps=10,
    #         val_check_interval=config.eval_every,
    #         devices=1)

    tic = time.perf_counter()
    trainer.fit(model, datamodule)
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    wandb.login()
    wandb.init(project="finetuning_molformer")#, id = "BACE_ring_count")
    main()
    wandb.finish()