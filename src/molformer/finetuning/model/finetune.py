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

    margs = AttributeDict(get_argparse_defaults(ARGS()))
    margs.device = 'cuda'
    margs.batch_size = 48
    margs.n_head = 12
    margs.n_layer = 12
    margs.n_embd = 768
    margs.d_dropout = 0.1
    margs.dropout = 0.1
    margs.lr_start = 3e-5
    margs.num_workers = 8
    margs.max_epochs = 50
    margs.num_feats = 32
    margs.seed_path = '../../../../../molformer/data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
    margs.dataset_name = 'bace'
    margs.data_root = '../../../../data/bace'
    margs.measure_name = 'Class'
    margs.dims = [768, 768, 768, 1]
    margs.checkpoints_folder = './checkpoints_bace'
    margs.num_classes = 2


    print("Using " + str(
        torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    pos_emb_type = 'rot'
    print('pos_emb_type is {}'.format(pos_emb_type))

    run_name_fields = [
        margs.dataset_name,
        margs.measure_name,
        pos_emb_type,
        margs.fold,
        margs.mode,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
        margs.dims,
    ]

    run_name = "_".join(map(str, run_name_fields))

    print("RUN", run_name)
    arguments = dict(margs)
    datamodule = PropertyPredictionDataModule(arguments)
    margs.dataset_names = "valid test".split()
    margs.run_name = run_name

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder)
    margs.checkpoint_root = checkpoint_root
    margs.run_id=np.random.randint(30000)
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "models_"+str(margs.run_id))
    results_dir = os.path.join(checkpoint_root, "results")
    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    # os.makedirs(results_dir, exist_ok=True)
    # os.makedirs(checkpoint_dir, exist_ok=True)
    

    # checkpoint_path = os.path.join(checkpoints_folder, margs.measure_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k = -1, 
                                                       save_last = True,
                                                       dirpath=checkpoint_dir, 
                                                       filename='checkpoint', 
                                                       verbose=True)


    tokenizer = MolTranBertTokenizer('../../data/bert_vocab.txt')
    seed_everything(margs.seed)

    if margs.seed_path == '':
        print("# training from scratch")
        model = LightningModule(margs, tokenizer)
    else:
        print(f"# loaded pre-trained model from {margs.seed_path}")
        model = LightningModule(margs, tokenizer).load_from_checkpoint(margs.seed_path, 
                                                                       strict=False,    
                                                                       config=margs, 
                                                                       tokenizer=tokenizer, 
                                                                       vocab=len(tokenizer.vocab))

    last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
    
    resume_from_checkpoint = None
    
    if os.path.isfile(last_checkpoint_file):
        print(f"resuming training from : {last_checkpoint_file}")
        resume_from_checkpoint = last_checkpoint_file
    else:
        print(f"training from saved")

    trainer = pl.Trainer(
        max_epochs=margs.max_epochs,
        log_every_n_steps=10,
        default_root_dir=checkpoint_root,
        accelerator= "gpu",
        devices=1,
        callbacks =  [checkpoint_callback],
        enable_checkpointing= True,
        num_sanity_val_steps=5,
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
    trainer.fit(model, datamodule, ckpt_path=resume_from_checkpoint)
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    wandb.login()
    wandb.init(project="clean_finetune", id = "BACE_BATCH_SIZE_48")
    main()
    wandb.finish()