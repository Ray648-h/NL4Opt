import os
import sys, random
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer
from config import Config
from model import TextMappingModel
from data_per_declaration import DeclarationMappingDataset
from constants import SPECIAL_TOKENS
from utils import *
import test_utils

CKPT = '/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/output/baseline/20261116_153103/best-checkpoint.mdl'
gpu = 0
use_gpu = True

checkpoint = torch.load(CKPT, map_location='cuda:0')
config = Config.from_dict(checkpoint['config'])
config.gpu_device = gpu
config.use_gpu = use_gpu
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False
torch.cuda.set_device(config.gpu_device)

model_name = config.bert_model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.bert_cache_dir)
tokenizer.add_tokens(SPECIAL_TOKENS)

vocabs = {}
dev_set = DeclarationMappingDataset(config.dev_file, max_length=config.max_length, gpu=use_gpu)
dev_set.numberize(tokenizer, vocabs)

model = TextMappingModel(config, vocabs)
model.load_bert(model_name, cache_dir=config.bert_cache_dir, tokenizer=tokenizer)
model.bert.resize_token_embeddings(len(tokenizer))
model.load_state_dict(checkpoint['model'], strict=True)
model.cuda(device=config.gpu_device)

batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
print(f'dev set size = {len(dev_set)}, batch_num = {batch_num}')

for nb in [1, 3, 5, 8]:
    model.num_beams = nb
    result = test_utils.evaluate(tokenizer, model, dev_set, 1000, batch_num,
                                 use_gpu, config, tqdm_descr=f'Dev b{nb}', print_errors=False)
    print(f'num_beams={nb}  ->  dev accuracy = {result["accuracy"]:.4f}', flush=True)
