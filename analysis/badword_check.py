import os
import sys
import torch
from types import SimpleNamespace

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir("/home/ray/projects/nl4opt/nl4opt-subtask2-baseline")
sys.path.insert(0, os.getcwd())

from transformers import AutoTokenizer, AutoConfig, LogitsProcessorList, NoBadWordsLogitsProcessor
from model import TextMappingModel
from config import Config
from constants import SPECIAL_TOKENS

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir="./bert")
tokenizer.add_tokens(SPECIAL_TOKENS)

print("=== 1. token ids ===")
print("bos:", tokenizer.bos_token_id, "| eos:", tokenizer.eos_token_id,
      "| sep:", tokenizer.sep_token_id, "| pad:", tokenizer.pad_token_id)
cfg = AutoConfig.from_pretrained("facebook/bart-base", cache_dir="./bert")
print("config.decoder_start_token_id:", cfg.decoder_start_token_id)

print("\n=== 2. Processor 构造 vs 实际生效的禁止序列 ===")
bad_words = [[tokenizer.sep_token_id], [tokenizer.pad_token_id]]
print("传入的 bad_words:", bad_words)
proc = NoBadWordsLogitsProcessor(bad_words_ids=bad_words, eos_token_id=tokenizer.eos_token_id)
print("proc.bad_word_ids (原始):", proc.bad_word_ids)
print("proc.sequence_bias (实际 -inf 的序列):", getattr(proc, "sequence_bias", None))

print("\n=== 3. 单步单元测试：EOS 是否真被禁 ===")
scores = torch.zeros(1, len(tokenizer))
input_ids = torch.tensor([[tokenizer.bos_token_id]])
filt = proc(input_ids, scores)
for name, tid in [("PAD", tokenizer.pad_token_id), ("EOS/SEP", tokenizer.eos_token_id), ("BOS", tokenizer.bos_token_id)]:
    print(f"  {name}(id={tid}): 原始={scores[0, tid].item():.2f} -> 过滤后={filt[0, tid].item():.2f}")

print("\n=== 4. 多 token 禁止序列测试 ===")
mproc = NoBadWordsLogitsProcessor(bad_words_ids=[[tokenizer.bos_token_id, tokenizer.eos_token_id]],
                                  eos_token_id=tokenizer.eos_token_id)
s = torch.zeros(1, len(tokenizer))
v1 = mproc(torch.tensor([[tokenizer.bos_token_id]]), s.clone())[0, tokenizer.eos_token_id].item()
v2 = mproc(torch.tensor([[tokenizer.bos_token_id, tokenizer.pad_token_id]]), s.clone())[0, tokenizer.eos_token_id].item()
print(f"  前缀=[bos] 禁 [bos,eos] -> EOS 分数={v1} (应为 -inf，bos 是前缀)")
print(f"  前缀=[bos,pad] 禁 [bos,eos] -> EOS 分数={v2} (应为 0，前缀不匹配)")

print("\n=== 5. 真实 generate() 中逐步 EOS/PAD 分数 + 生成 id ===")
config = Config(bert_model_name="facebook/bart-base", bert_cache_dir="./bert", use_copy=True, k=10,
                max_position_embeddings=1024, max_length=512, num_beams=1, use_extra_bert=False,
                bert_dropout=0.0, SOT_weights=5, decoding_method="greedy",
                use_gpu=torch.cuda.is_available(), gpu_device=0, seed=0)
model = TextMappingModel(config, vocabs={})
model.load_bert("facebook/bart-base", cache_dir="./bert", tokenizer=tokenizer)
model.bert.resize_token_embeddings(len(tokenizer))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

text = "minimize the cost of shipping the goods from factory to store"
enc = tokenizer(text, return_tensors="pt")
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

# 与 model.py generate() 完全一致的构造
logits_processor = LogitsProcessorList(
    [NoBadWordsLogitsProcessor(
        bad_words_ids=[[tokenizer.sep_token_id], [tokenizer.pad_token_id]],
        eos_token_id=tokenizer.eos_token_id)]
)
model.bert._cache_input_ids = input_ids

with torch.no_grad():
    out = model.bert.generate(
        input_ids, attention_mask=attention_mask, max_length=10, num_beams=1,
        logits_processor=logits_processor, return_dict_in_generate=True, output_scores=True,
    )

ids = out.sequences[0].tolist()
print("生成 ids:", ids)
print("逐 token:", [tokenizer.convert_ids_to_tokens(i) for i in ids])
print("位置0 =", tokenizer.convert_ids_to_tokens(ids[0]), "(=decoder_start 则是初始输入，非生成)")
print("是否出现 PAD(1):", 1 in ids, "| 是否出现 EOS(2):", 2 in ids)
for step, sc in enumerate(out.scores):
    print(f"  step {step}: EOS(2)={sc[0, tokenizer.eos_token_id].item():.4f}  PAD(1)={sc[0, tokenizer.pad_token_id].item():.4f}")
