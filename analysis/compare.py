import torch
old = torch.load("/home/ray/projects/nl4opt/_old_out.pt", weights_only=False)
new = torch.load("/home/ray/projects/nl4opt/_new_out.pt", weights_only=False)

def stats(a, b):
    a = a.float(); b = b.float()
    if a.numel() == 0: return 0.0, 0.0
    absd = (a - b).abs().max().item()
    denom = b.abs().clamp_min(1e-12)
    reld = ((a - b).abs() / denom).max().item()
    return absd, reld

print("=== 5.1 前向 + 5.2 反向 ===")
keys = ["loss", "encoder_last_hidden", "decoder_last_hidden", "final_logits",
        "cross_attn_last", "copy_logits", "p_gen", "raw_lm_logits"]
for k in keys:
    a, b = old[k], new[k]
    if isinstance(a, torch.Tensor) and a.numel() == 1:
        absd, reld = stats(a.reshape(1), b.reshape(1))
    else:
        absd, reld = stats(a, b)
    try:
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        verdict = "PASS"
    except Exception as e:
        verdict = "FAIL: " + str(e).split("\n")[0][:80]
    print(f"  {k:22s} max_abs={absd:.3e} max_rel={reld:.3e}  {verdict}")

print("  selected_heads equal:", bool((old["selected_heads"] == new["selected_heads"]).all()),
      "| final_logits_bias 同为 buffer/param:", old["final_logits_bias_is_param"] == new["final_logits_bias_is_param"],
      "(", old["final_logits_bias_is_param"], ")")

print("  --- 梯度 ---")
for k in old["grads"]:
    a, b = old["grads"][k], new["grads"][k]
    if a is None or b is None:
        print(f"  grad {k}: old={a is not None} new={b is not None}")
        continue
    absd, reld = stats(a, b)
    try:
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        v = "PASS"
    except Exception as e:
        v = "FAIL: " + str(e).split("\n")[0][:80]
    print(f"  grad {k:55s} max_abs={absd:.3e} max_rel={reld:.3e}  {v}")

print("\n=== 5.3 生成 ===")
print("  greedy old :", old["greedy_ids"].tolist())
print("  greedy new :", new["greedy_ids"].tolist())
print("  greedy 完全一致:", bool((old["greedy_ids"] == new["greedy_ids"]).all()))
print("  beam old   :", old["beam_ids"].tolist(), "|", old["beam_note"])
print("  beam new   :", new["beam_ids"].tolist(), "|", new["beam_note"])
# 去掉尾部 pad 后，old top-1 vs new 最优束
oldb = old["beam_ids"][0].tolist()
newb = new["beam_ids"][0].tolist()
while oldb and oldb[-1] == 1: oldb.pop()
while newb and newb[-1] == 1: newb.pop()
print("  beam top-1(去尾部pad) old==new:", oldb == newb, "| old:", oldb, "| new:", newb)
