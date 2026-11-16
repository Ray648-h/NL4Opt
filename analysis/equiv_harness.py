import os, sys, torch, torch.nn.functional as F
from types import SimpleNamespace
from collections import OrderedDict

repo_dir = sys.argv[1]
out_path = sys.argv[2]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

from transformers import AutoTokenizer
from model import TextMappingModel
from config import Config

CACHE = "/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/bert"

torch.manual_seed(0)
torch.backends.cudnn.enabled = False

tok = AutoTokenizer.from_pretrained("facebook/bart-base", cache_dir=CACHE)   # 不加 SPECIAL_TOKENS，不 resize
cfg = Config(bert_model_name="facebook/bart-base", bert_cache_dir=CACHE, use_copy=True, k=10,
             max_position_embeddings=1024, max_length=512, num_beams=1, use_extra_bert=False,
             bert_dropout=0.0, SOT_weights=5, decoding_method="greedy",
             use_gpu=torch.cuda.is_available(), gpu_device=0, seed=0)
model = TextMappingModel(cfg, vocabs={})
model.load_bert("facebook/bart-base", cache_dir=CACHE, tokenizer=tok)
model.bert.config.output_attentions = True   # 原代码 assert 需要；迁移版已在 forward 内强制，这里无副作用
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
model.eval()   # 关 dropout，保证前后向确定；梯度仍会流动
V = len(tok)

src_text = "minimize the total shipping cost"
tgt_text = "minimize the total shipping cost"
enc = tok(src_text, return_tensors="pt").to(dev)
dec = tok(tgt_text, return_tensors="pt")["input_ids"].to(dev)
labels = dec.clone()

# ===== 5.1 前向 + 5.2 反向（一次前向，eval 模式） =====
model.bert._loss_weight = torch.ones(V, device=dev)
model.bert._vocab_size = V
model.bert._cache_input_ids = enc["input_ids"]
out = model.bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 decoder_input_ids=dec, labels=labels, return_dict=True, output_hidden_states=True)

res = OrderedDict()
res["loss"] = out.loss.detach().cpu()
res["encoder_last_hidden"] = out.encoder_last_hidden_state.detach().cpu()
res["decoder_last_hidden"] = out.decoder_hidden_states[-1].detach().cpu()
res["final_logits"] = out.logits.detach().cpu()
res["cross_attn_last"] = out.cross_attentions[-1].detach().cpu()
res["selected_heads"] = model.bert.selected_heads.detach().cpu()

sel = model.bert.selected_heads
agg = out.cross_attentions[-1][:, sel, :, :].mean(1)
dummy = enc["input_ids"].unsqueeze(-1).expand(-1, -1, out.logits.size(1)).transpose(1, 2)
copy_logits = torch.zeros_like(out.logits); copy_logits.scatter_add_(2, dummy, agg)
dec_h = out.decoder_hidden_states[-1]; enc_h = out.encoder_last_hidden_state
p_gen = torch.sigmoid(torch.bmm(dec_h, enc_h.mean(1).unsqueeze(-1)))
raw_lm = model.bert.lm_head(dec_h) + model.bert.final_logits_bias
res["copy_logits"] = copy_logits.detach().cpu()
res["p_gen"] = p_gen.detach().cpu()
res["raw_lm_logits"] = raw_lm.detach().cpu()
res["final_logits_bias_is_param"] = any(n == "final_logits_bias" for n, _ in model.bert.named_parameters())

# 5.2 反向
model.zero_grad()
out.loss.backward()
grads = OrderedDict()
keys = ["lm_head.weight", "shared.weight", "encoder.layers.0.self_attn.q_proj.weight",
        "decoder.layers.5.encoder_attn.q_proj.weight", "decoder.layers.5.encoder_attn.out_proj.weight"]
for n, p in model.bert.named_parameters():
    if any(k in n for k in keys):
        grads[n] = p.grad.detach().cpu() if p.grad is not None else None
res["grads"] = grads

# ===== 5.3 生成：贪心（num_beams=1，走各自 TextMappingModel.generate） =====
batch = SimpleNamespace(input_ids=enc["input_ids"], attention_masks=enc["attention_mask"])
model.bert._cache_input_ids = enc["input_ids"]
with torch.no_grad():
    gids = model.generate(batch, num_beams=1, decoding_length=14, tokenizer=tok)
res["greedy_ids"] = gids.cpu()

# ===== 5.3 生成：beam（num_beams=3） =====
try:
    if hasattr(model.bert, "beam_search"):
        # 旧代码：beam_search + 修正后的 cache（原代码 _cache_input_ids 未扩展，num_beams>1 会崩）
        nb = 3
        model.bert._cache_input_ids = enc["input_ids"].repeat_interleave(nb, dim=0)
        enc_out = model.bert.get_encoder()(enc["input_ids"].repeat_interleave(nb, dim=0),
                                           enc["attention_mask"].repeat_interleave(nb, dim=0), return_dict=True)
        dec_ids = torch.ones((nb * enc["input_ids"].size(0), 1), device=dev, dtype=torch.long) * model.bert.config.decoder_start_token_id
        from transformers.generation.utils import BeamSearchScorer
        scorer = BeamSearchScorer(batch_size=enc["input_ids"].size(0), max_length=14, num_beams=nb, device=dev)
        from transformers import NoBadWordsLogitsProcessor
        lp = NoBadWordsLogitsProcessor(bad_words_ids=[[tok.sep_token_id], [tok.pad_token_id]], eos_token_id=tok.eos_token_id)
        with torch.no_grad():
            bids = model.bert.beam_search(dec_ids, scorer, max_length=14, logits_processor=lp, encoder_outputs=enc_out)
        res["beam_ids"] = bids.cpu()
        res["beam_note"] = "old beam_search + cache fix"
    else:
        nb = 3
        model.bert._cache_input_ids = enc["input_ids"].repeat_interleave(nb, dim=0)
        from transformers import LogitsProcessorList, NoBadWordsLogitsProcessor
        lp = LogitsProcessorList([NoBadWordsLogitsProcessor(bad_words_ids=[[tok.sep_token_id], [tok.pad_token_id]], eos_token_id=tok.eos_token_id)])
        with torch.no_grad():
            bids = model.bert.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                       max_length=14, num_beams=nb, num_return_sequences=nb, logits_processor=lp)
        res["beam_ids"] = bids.cpu()
        res["beam_note"] = "new generate beam"
except Exception as e:
    res["beam_ids"] = None
    res["beam_note"] = f"CRASH: {type(e).__name__}: {e}"

torch.save(dict(res), out_path)
print("saved ->", out_path)
print("  loss:", float(res["loss"]), "| greedy:", res["greedy_ids"].tolist())
print("  beam_note:", res["beam_note"], "| beam:", res["beam_ids"].tolist() if res["beam_ids"] is not None else None)
print("  grads:", {k: (None if v is None else float(v.abs().max())) for k, v in res["grads"].items()})
