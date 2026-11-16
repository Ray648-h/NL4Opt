import os, sys, torch, torch.nn.functional as F
from collections import Counter
from types import SimpleNamespace
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, "/home/ray/projects/nl4opt/nl4opt-subtask2-baseline")
os.chdir("/home/ray/projects/nl4opt/nl4opt-subtask2-baseline")

from transformers import AutoTokenizer
from model import TextMappingModel
from config import Config
from constants import SPECIAL_TOKENS

tok = AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir="./bert")
tok.add_tokens(SPECIAL_TOKENS)
cfg = Config(bert_model_name="facebook/bart-base", bert_cache_dir="./bert", use_copy=True, k=10,
             max_position_embeddings=1024, max_length=512, num_beams=1, use_extra_bert=False,
             bert_dropout=0.0, SOT_weights=5, decoding_method="greedy",
             use_gpu=torch.cuda.is_available(), gpu_device=0, seed=0)
model = TextMappingModel(cfg, vocabs={})
model.load_bert("facebook/bart-base", cache_dir="./bert", tokenizer=tok)
model.bert.resize_token_embeddings(len(tok))
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev); model.eval()
V = len(tok)

print("final_logits_bias 是参数还是 buffer?",
      "param" if any(n=="final_logits_bias" for n,_ in model.bert.named_parameters()) else "buffer")

src_text = "the the cost of shipping"
enc = tok(src_text, return_tensors="pt").to(dev)
dec = tok("minimize the total cost", return_tensors="pt")["input_ids"].to(dev)

model.bert._loss_weight = torch.ones(V, device=dev)
model.bert._vocab_size = V
model.bert._cache_input_ids = enc["input_ids"]

with torch.no_grad():
    out = model.bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                     decoder_input_ids=dec, return_dict=True, output_hidden_states=True)

final = out.logits
print("\n== 检查3: 最终分布 ==")
print("  shape:", tuple(final.shape), "| finite:", torch.isfinite(final).all().item(),
      "| >=0:", (final >= 0).all().item())
print("  每步 sum(应≈1):", final.sum(-1).tolist())

sel = model.bert.selected_heads
last_ca = out.cross_attentions[-1]
agg = last_ca[:, sel, :, :].mean(1)
dummy = enc["input_ids"].unsqueeze(-1).expand(-1, -1, final.size(1)).transpose(1, 2)
copy_logits = torch.zeros_like(final)
copy_logits.scatter_add_(2, dummy, agg)

print("\n== 检查1: copy 分布 ==")
print("  agg 形状:", tuple(agg.shape), "| copy_logits 形状:", tuple(copy_logits.shape))
print("  copy_logits 每步 sum(应≈1):", copy_logits.sum(-1).tolist())
cnt = Counter(enc["input_ids"][0].tolist())
dup = [t for t, c in cnt.items() if c >= 2]
print("  输入中重复 token id:", dup, "->", tok.convert_ids_to_tokens(dup))
t = 0
for v in dup:
    pos = (enc["input_ids"][0] == v).nonzero(as_tuple=True)[0]
    manual = agg[0, t, pos].sum()
    actual = copy_logits[0, t, v]
    print(f"  id={v}: scatter_add={actual.item():.6f} vs 位置注意力求和={manual.item():.6f} (应相等)")
absent = tok.convert_tokens_to_ids("factory")
print(f"  源里不存在的 factory(id={absent}) copy 质量={copy_logits[0,:,absent].max().item():.2e} (应为0)")

dec_h = out.decoder_hidden_states[-1]
enc_h = out.encoder_last_hidden_state
p_gen = torch.sigmoid(torch.bmm(dec_h, enc_h.mean(1).unsqueeze(-1)))
print("\n== 检查2: p_gen ==")
print("  p_gen 形状:", tuple(p_gen.shape), "| min:", p_gen.min().item(), "| max:", p_gen.max().item())

raw_lm = model.bert.lm_head(dec_h) + model.bert.final_logits_bias
fused = F.softmax(raw_lm, -1) * p_gen + copy_logits * (1 - p_gen)
print("  重算融合 vs 模型 logits 最大绝对差:", (fused - final).abs().max().item(), "(应≈0)")

print("\n== 检查4: 梯度 ==")
model.train()
src2 = tok("minimize the total shipping cost", return_tensors="pt").to(dev)
dec2 = tok("minimize the total shipping cost", return_tensors="pt")["input_ids"].to(dev)
b = SimpleNamespace(input_ids=src2["input_ids"], attention_masks=src2["attention_mask"])
model.bert._loss_weight = torch.ones(V, device=dev)
model.bert._vocab_size = V
res = model(b, dec2, dec2.clone(), tokenizer=tok)
loss = res["loss"]
print("  loss:", loss.item(), "| finite:", torch.isfinite(loss).item())
model.zero_grad()
loss.backward()
targets = ["lm_head.weight", "final_logits_bias",
           "model.encoder.layers.0.self_attn.q_proj.weight",
           "model.decoder.layers.11.encoder_attn.out_proj.weight",
           "model.decoder.layers.11.encoder_attn.q_proj.weight"]
for name in targets:
    try:
        p = model.bert.get_parameter(name)
    except Exception:
        print(f"  {name}: 不存在"); continue
    if not p.requires_grad:
        print(f"  {name}: requires_grad=False (buffer)"); continue
    g = p.grad
    if g is None:
        print(f"  {name}: grad=None")
    else:
        print(f"  {name}: finite={torch.isfinite(g).all().item()} nonzero={(g!=0).any().item()} absmax={g.abs().max().item():.6f}")
