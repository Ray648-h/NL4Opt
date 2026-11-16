import os
import sys, json, logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import parsers, scoring

res = json.load(open('/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/output/baseline/20261102_141917/result.dev.json_199'))
gold_texts = res['gold_texts']
pred_texts = res['pred_texts']
doc_ids = res['doc_ids']

# Replicate collate_score_declarations grouping (contiguous doc_ids -> one problem)
def group(texts, doc_ids):
    problems, cur, cur_id = [], [], None
    for t, d in zip(texts, doc_ids):
        if cur_id is not None and d != cur_id:
            problems.append(''.join(cur)); cur = []
        cur_id = d; cur.append(t)
    if cur: problems.append(''.join(cur))
    return problems

gold_problems = group(gold_texts, doc_ids)
pred_problems = group(pred_texts, doc_ids)

print('number of problems:', len(gold_problems))
for i, (gp, pp) in enumerate(zip(gold_problems, pred_problems)):
    print('=' * 90)
    print('PROBLEM', i)
    parser = parsers.ModelOutputXMLParser(print_errors=False)

    g = parser.parse(gp)
    p = parser.parse(pp)

    print(f'  GOLD -> obj.direction={g.objective.direction!r} obj.name={g.objective.name!r} '
          f'n_obj_terms={len(g.objective.terms)} n_constraints={len(g.constraints)}')
    print(f'  PRED -> obj.direction={p.objective.direction!r} obj.name={p.objective.name!r} '
          f'n_obj_terms={len(p.objective.terms)} n_constraints={len(p.constraints)}')

    gc = parsers.convert_to_canonical(g)
    pc = parsers.convert_to_canonical(p)
    # sanity: scoring gold-vs-itself must be 1.0 if scoring works
    self_score = scoring.overall_score([gc.objective], [gc.constraints], [gc.objective], [gc.constraints])
    pred_score = scoring.overall_score([pc.objective], [pc.constraints], [gc.objective], [gc.constraints])
    print(f'  SCORING sanity: score(gold, gold)={self_score:.4f}   score(pred, gold)={pred_score:.4f}')
