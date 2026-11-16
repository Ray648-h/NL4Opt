import os
import json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

res = json.load(open('/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/output/baseline/20261116_153103/result.dev.json_19'))
preds = res['pred_texts']
golds = res['gold_texts']
docs = res['doc_ids']

print(f'total declarations = {len(preds)}, accuracy = {res["accuracy"]}')

mismatch = [(p, g, d) for p, g, d in zip(preds, golds, docs) if p.strip() != g.strip()]
exact_match = len(preds) - len(mismatch)
print(f'exact string matches = {exact_match}, mismatches = {len(mismatch)}')

print('\n================ FIRST 25 MISMATCHES ================')
for i, (p, g, d) in enumerate(mismatch[:25]):
    print(f'\n--- decl {i} (doc {d}) ---')
    print(f'  GOLD: {g}')
    print(f'  PRED: {p}')
