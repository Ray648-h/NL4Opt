import os
import json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import parsers

res = json.load(open('/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/output/baseline/20261116_153103/result.dev.json_19'))
pairs = res['gold_pred_pairs']

def fmt(f):
    o = f.objective
    obj = f"dir={o.direction!r} name={o.name!r} terms={[(k, v.value) for k, v in o.terms.items()]}"
    cons = []
    for c in f.constraints:
        cons.append(f"type={c.type!r} dir={c.direction!r} op={c.operator!r} limit={c.limit!r} terms={[(k, v.value) for k, v in c.terms.items()]}")
    return f"OBJ[{obj}] CONS{cons}"

# build order mapping from gold for alignment (mirror real scoring)
def build_mapping(gold_text):
    gp = parsers.ModelOutputXMLParser(print_errors=False)
    g = gp.parse(gold_text, None)
    return g.entities

shown = 0
for i, pair in enumerate(pairs):
    g, p = pair['gold'], pair['pred']
    if g.strip() == p.strip():
        continue
    om = build_mapping(g)
    parser = parsers.ModelOutputXMLParser(print_errors=False)
    gf = parser.parse(g, om)
    pf = parser.parse(p, om)
    gs, ps = fmt(gf), fmt(pf)
    same = (gs == ps)
    tag = 'FORGIVEN (parse same)' if same else 'REALLY WRONG'
    print(f'\n--- decl {i} [{tag}] ---')
    print(f'  GOLD parse: {gs}')
    print(f'  PRED parse: {ps}')
    shown += 1
    if shown >= 15:
        break
