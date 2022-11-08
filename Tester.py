import torch
import torch.nn.functional as F
import torch.utils.data
from torch import cuda
import opts
from Datamodule import APETestDataset
from apex.fp16_utils import *
from tqdm import tqdm
import wandb
import os

from mbart_modeling import MBartForConditionalGeneration
from transformers import MBart50TokenizerFast, MBartConfig
import nltk
import nltk.translate.bleu_score as bleu

import os

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = opts.get_args()

args.local_rank = os.getenv('LOCAL_RANK', 0)

args.ckpt_path = f"{args.ckpt_path}/{args.experiment_name}"
print("Check point path : {}".format(args.ckpt_path))
if not os.path.isdir(args.ckpt_path):
    print(">>>>>>Wrong checkpoint!!")
    os.mkdir(args.ckpt_path)

torch.manual_seed(args.seed)

# FOR DISTRIBUTED:  Set the device according to local_rank.
args.gpu = int(args.local_rank)
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend='nccl',
                                        init_method='env://')
args.world_size = torch.distributed.get_world_size()
torch.backends.cudnn.benchmark = True

tokenizer = MBart50TokenizerFast.from_pretrained(args.ckpt_path)

config = MBartConfig.from_pretrained(args.ckpt_path)



model = MBartForConditionalGeneration.from_pretrained(args.ckpt_path, config = config)

model = model.to(device)
    
print("Model Setup Ready...")


test_dataset = APETestDataset(args,
                               path = args.data_path,      
                               tokenizer = tokenizer
                               )

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = 1, 
                                          shuffle=False, 
                                          pin_memory=True)
bleu_score = 0.0
model.eval()
with torch.no_grad():
    with open(f"outputs/{args.experiment_name}_outputs.txt", "w") as o:
        with open(f"outputs/{args.experiment_name}_labels.txt", "w") as l:
            for step, test_batch in tqdm(enumerate(test_loader), desc=f"{args.experiment_name} inference..."):
                
                test_batch = tuple(input_tensor.to(device) for input_tensor in test_batch.values())
                input_ids, decoder_input_ids, labels = test_batch
                
                outputs =  model.generate(input_ids, num_beams=4)
                output = tokenizer.decode(outputs[0])

                label_txt = tokenizer.decode(labels[0])
                bleu_score += bleu.sentence_bleu([output], label_txt)
                o.write(f"{output}\n")
                l.write(f"{label_txt}\n")

bleu_score = bleu_score / test_dataset.__len__()
print(f"****************BLEU SCORE : {bleu_score}*************8")