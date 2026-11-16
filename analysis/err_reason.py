import os
import json, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import parsers

res = json.load(open('/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/output/baseline/20261116_153103/result.dev.json_19'))
preds, golds, docs = res['pred_texts'], res['gold_texts'], res['doc_ids']

def group(texts, doc_ids):
    problems, cur, cur_id = [], [], None
    for t, d in zip(texts, doc_ids):
        if cur_id is not None and d != cur_id:
            problems.append(''.join(cur)); cur = []
        cur_id = d; cur.append(t)
    if cur: problems.append(''.join(cur))
    return problems

gold_probs = group(golds, docs)
pred_probs = group(preds, docs)

def parse(text):
    p = parsers.ModelOutputXMLParser(print_errors=False)
    return p.parse(text, None)

# categorize at DECLARATION level (align objective, then constraints by order)
err_counter = Counter()
total_decl = 0

for gtxt, ptxt in zip(gold_probs, pred_probs):
    g = parse(gtxt)
    p = parse(ptxt)

    # objective
    go, po = g.objective, p.objective
    total_decl += 1
    if go.direction != po.direction:
        err_counter['obj direction wrong'] += 1
    if len(go.terms) != len(po.terms):
        err_counter['obj term count mismatch (truncation/extra)'] += 1
    else:
        for (gn, gv), (pn, pv) in zip(sorted(go.terms.items()), sorted(po.terms.items())):
            if gn != pn:
                err_counter['obj entity wrong'] += 1
            elif gv.value != pv.value:
                err_counter['obj value wrong'] += 1

    # constraints (align by order)
    gc, pc = g.constraints, p.constraints
    for i in range(max(len(gc), len(pc))):
        total_decl += 1
        if i >= len(gc):
            err_counter['constraint extra (pred has more)'] += 1; continue
        if i >= len(pc):
            err_counter['constraint missing (pred truncated)'] += 1; continue
        c1, c2 = gc[i], pc[i]
        if c1.type != c2.type:
            err_counter['constraint type wrong'] += 1
        if c1.direction != c2.direction:
            err_counter['constraint direction wrong'] += 1
        if c1.operator != c2.operator:
            err_counter['constraint operator wrong'] += 1
        if c1.limit != c2.limit:
            err_counter['constraint limit wrong'] += 1
        if len(c1.terms) != len(c2.terms):
            err_counter['constraint term count mismatch'] += 1
        else:
            for (gn, gv), (pn, pv) in zip(sorted(c1.terms.items()), sorted(c2.terms.items())):
                if gn != pn:
                    err_counter['constraint entity wrong'] += 1
                elif gv.value != pv.value:
                    err_counter['constraint value wrong'] += 1

print(f'total declarations compared = {total_decl}')
print('\n=== real (content/structural) error counts ===')
for k, v in err_counter.most_common():
    print(f'{v:4d}  {k}')
