import os
import json, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import parsers, scoring

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

def canon(text, order_mapping=None):
    p = parsers.ModelOutputXMLParser(print_errors=False)
    parsed = p.parse(text, order_mapping)
    return parsers.convert_to_canonical(parsed), parsed

def score(pred_texts, gold_texts):
    po, pc, to, tc = [], [], [], []
    for gt, pt in zip(gold_texts, pred_texts):
        gc, gp = canon(gt, None)
        om = gp.entities
        pc2, _ = canon(pt, om)
        to.append(gc.objective); tc.append(gc.constraints)
        po.append(pc2.objective); pc.append(pc2.constraints)
    return scoring.overall_score(po, pc, to, tc)

base = score(pred_probs, gold_probs)
print(f'reimplemented accuracy (fuzzy) = {base:.4f}   (official = {res["accuracy"]})')

def cleanup(t):
    t = re.sub(r'</?\s*OBJ_+NAME', lambda m: m.group(0).replace('__', '_').replace(' ', ''), t)
    t = re.sub(r'<PARAM\s+>', '<PARAM>', t)
    t = re.sub(r'</\s+PARAM', '</PARAM', t)
    t = re.sub(r'</?(ParAM|PARam|Param)(?=>|\s)', lambda m: m.group(0).upper(), t)
    t = re.sub(r'</PARAM>\.', '</PARAM>', t)
    t = re.sub(r'\[TI[MPSA]+?\]', '[TIMES]', t)
    t = re.sub(r'[\(\[]TIMES[\]\)]', '[TIMES]', t)
    return t

clean_preds = [cleanup(p) for p in pred_probs]
clean = score(clean_preds, gold_probs)
print(f'after surface cleanup           = {clean:.4f}   (delta {clean-base:+.4f})')
