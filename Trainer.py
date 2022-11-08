import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
from torch import cuda
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
# from cococaption.pycocotools.coco import COCO
# from cococaption.pycocoevalcap.eval import COCOEvalCap
from PIL import Image
import opts
from Datamodule import APETrainDataset, APEValDataset, APETestDataset
from apex import amp
from apex.fp16_utils import *
from tqdm import tqdm
import wandb
import os
from apex.parallel import DistributedDataParallel
from mbart_modeling import MBartForConditionalGeneration
from transformers import MBart50TokenizerFast, MBartConfig
import pandas as pd

def change_requires_grad(model, req_grad):
    for n,p in model.named_parameters():
        if "adapter_layer" in n:
            continue
        p.requires_grad = req_grad
    
def save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
        
    # model.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)
        
    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
            **kwargs}
    torch.save(opt, ckpt_path + filename)
    model.module.save_pretrained(ckpt_path)

def get_optimizer(model, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = opts.get_args()

print(f"adapter mode : {args.adapter_latent_size is not None}")
print(f"train_mode : {args.train_mode}")


args.local_rank = os.getenv('LOCAL_RANK', 0)

torch.manual_seed(args.seed)
wandb.init(project = args.project_name, name = args.experiment_name)
wandb.config.update(args)
wandb.run.name = args.experiment_name

# FOR DISTRIBUTED:  Set the device according to local_rank.
args.gpu = int(args.local_rank)
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend='nccl',
                                        init_method='env://')
args.world_size = torch.distributed.get_world_size()
torch.backends.cudnn.benchmark = True

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

special_tokens_lst = tokenizer.additional_special_tokens
special_tokens_lst.append("ape_kor")
tokenizer.add_special_tokens({"additional_special_tokens":special_tokens_lst})
# num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
#                                                'additional_special_tokens': ['<question>', '<answer>', '<explanation>']})

# assert len(tokenizer) == orig_num_tokens + num_new_tokens
config = MBartConfig.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Add configs
config.add_cross_attention = True
config.adapter_latent_size = args.adapter_latent_size
config.adapter_non_linearity = "relu"
config.adapter_residual = True


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", config = config)


if config.adapter_latent_size is not None:
    change_requires_grad(model, False)

model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
wandb.watch(model)
optimizer = get_optimizer(model, args)
    
print("Model Setup Ready...")

train_dataset = APETrainDataset(
    args,
    path = args.data_path,
    tokenizer = tokenizer,
    )


val_dataset = APEValDataset(
    args,
    path = args.data_path,
    tokenizer = tokenizer,
    )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = args.train_batch_size, 
                                           shuffle=True, 
                                           pin_memory=True,
                                           drop_last = True,
                                           collate_fn=train_dataset.collate_fn
                                           )

val_loader =  torch.utils.data.DataLoader(val_dataset,
                                           batch_size = args.train_batch_size, 
                                           shuffle=True, 
                                           pin_memory=True,
                                           drop_last = True,
                                           collate_fn=val_dataset.collate_fn
                                           )


test_dataset = APETestDataset(args,
                               path = args.data_path,      
                               tokenizer = tokenizer
                               )

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = 1, 
                                          shuffle=False, 
                                          pin_memory=True)


# model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
if args.boost_mode == 'FP16':
    model = network_to_half(model)
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=128)
elif args.boost_mode == 'amp':
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

model = DistributedDataParallel(model)
    
t_total = (len(train_loader) // args.gradient_accumulation_steps) * args.num_train_epochs
warmup_steps = 0   # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

# if args.load_from_epoch is not None:
#     scheduler.load_state_dict(scheduler_dic)

args.ckpt_path = f"{args.ckpt_path}/{args.experiment_name}"
print("Check point path : {}".format(args.ckpt_path))
if not os.path.isdir(args.ckpt_path):
    os.mkdir(args.ckpt_path)

tokenizer.save_pretrained(args.ckpt_path)

for epoch in tqdm(range(0, args.num_train_epochs), desc = "Training mBART for APE"):
    
    model.train()
    accum_loss = 0
    accum_val_loss = 0
    prev_accum_val_loss = 10
    
    for step, (train_batch, val_batch) in enumerate(zip(train_loader, val_loader)):
        
        train_batch = tuple(input_tensor.to(device) for input_tensor in train_batch)
        input_ids, labels = train_batch
        outputs = model(input_ids=input_ids, 
                        past_key_values=None, 
                        labels=labels, 
                        use_cache=False, 
                        return_dict=True)

        train_loss = outputs["loss"]
        train_loss = train_loss / args.gradient_accumulation_steps
        with amp.scale_loss(train_loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        accum_loss += train_loss.item()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_batch = tuple(input_tensor.to(device) for input_tensor in val_batch)
            input_ids, labels = val_batch
            outputs = model(input_ids=input_ids, 
                        past_key_values=None, 
                        labels = labels,
                        use_cache=False, 
                        return_dict=True)
            val_loss = outputs["loss"]
            val_loss = val_loss / args.gradient_accumulation_steps
        accum_val_loss += val_loss.item()
        
        wandb.log({
            "Train loss" : train_loss,
            "Validation loss" : val_loss,
        })
        
        
        if step % args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print("\rEpoch {} / {}, Iter {} / {}, Train Loss: {:.3f}, Validation Loss: {:.3f}".format(
                epoch,
                args.num_train_epochs,
                step,
                len(train_loader),
                accum_loss,
                accum_val_loss
                ), end='          ')

            if accum_val_loss < prev_accum_val_loss:
                save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, args.ckpt_path)
                prev_accum_val_loss = accum_val_loss     

            accum_loss = 0
            accum_val_loss = 0
    