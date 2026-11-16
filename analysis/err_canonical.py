import os
import json, sys, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import parsers

res = json.load(open('/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/output/baseline/20261116_153103/result.dev.json_19'))
preds, golds, docs = res['pred_texts'], res['gold_texts'], res['doc_ids']

# --- group declarations into problems (contiguous doc_id), mirror real pipeline ---
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
print(f'problems = {len(gold_probs)}')

def canon(text, order_mapping=None):
    p = parsers.ModelOutputXMLParser(print_errors=False)
    parsed = p.parse(text, order_mapping)
    return parsers.convert_to_canonical(parsed), parsed

def arr_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))

def canon_equal(gc, pc):
    return arr_equal(gc.objective, pc.objective) and arr_equal(gc.constraints, pc.constraints)

correct = 0
for gp_text, pp_text in zip(gold_probs, pred_probs):
    gcanon, gparsed = canon(gp_text, None)          # build mapping from gold
    om = gparsed.entities
    pcanon, _ = canon(pp_text, om)                  # use gold's mapping on pred
    if canon_equal(gcanon, pcanon):
        correct += 1

print(f'REIMPLEMENTED problem-level accuracy = {correct}/{len(gold_probs)} = {correct/len(gold_probs):.4f}')
print(f'(official reported accuracy            = {res["accuracy"]})')

# --- surface cleanup: regex-fix known token corruptions, re-score, see if it helps ---
def cleanup(t):
    t = re.sub(r'</?\s*OBJ__+NAME', lambda m: m.group(0).replace('__', '_').replace(' ', ''), t)
    t = re.sub(r'<PARAM\s+>', '<PARAM>', t)
    t = re.sub(r'</\s+PARAM', '</PARAM', t)
    t = re.sub(r'</?(ParAM|PARam|Param)', lambda m: m.group(0).upper(), t)
    t = re.sub(r'</PARAM>\.', '</PARAM>', t)
    t = re.sub(r'\[(TI[MPSA]+?)\]', '[TIMES]', t)          # [TIMERS]/[TIPES]/[TAMES] -> [TIMES]
    t = re.sub(r'[\(\[]TIMES[\]\)]', '[TIMES]', t)
    return t

correct_clean = 0
for gp_text, pp_text in zip(gold_probs, pred_probs):
    gcanon, gparsed = canon(gp_text, None)
    om = gparsed.entities
    pcanon, _ = canon(cleanup(pp_text), om)
    if canon_equal(gcanon, pcanon):
        correct_clean += 1

print(f'AFTER surface cleanup                = {correct_clean}/{len(gold_probs)} = {correct_clean/len(gold_probs):.4f}')
print(f'-> cleanup gained {correct_clean - correct} problems')
