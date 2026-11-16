import json, re
from collections import Counter

res = json.load(open('/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/output/baseline/20261116_153103/result.dev.json_19'))
preds = res['pred_texts']
golds = res['gold_texts']

N = len(preds)
mis = sum(1 for p, g in zip(preds, golds) if p.strip() != g.strip())
print(f'total={N}  string-mismatch={mis}  exact-match={N-mis}')

# --- surface error detectors (pred contains the anomaly, gold does not) ---
def has(p, g, pat):
    return bool(re.search(pat, p)) and not bool(re.search(pat, g))

patterns = {
    'OBJ__NAME (double underscore tag)':     r'OBJ__+NAME',
    'OBJ_NAME wrong close (any _+ extra)':  r'</OBJ_+NAME',
    '[TIMES]->[TIMERS]/[TIPES]/[TAMES]':     r'\[TI[MPA][A-Z]*\]',
    'mismatched bracket (TIMERS] etc':      r'[\(\[]TI[MPA][A-Z]*[\]\)]',
    '<PARAM > space before >':              r'<PARAM\s+>',
    '</ PARAM> space after <':              r'</\s+PARAM',
    'wrong case ParAM/PARam/Param tag':     r'</?(?:ParAM|PARam|Param|PARAm)\b',
    'stray dot after </PARAM>':             r'</PARAM>\.',
    'stray quote in value':                 r'<PARAM[^>]*>"',
    'truncated decl (no </DECLARATION>)':   r'(?<!</DECLARATION>)\Z',
}

counts = Counter()
for p, g in zip(preds, golds):
    for name, pat in patterns.items():
        if has(p, g, pat):
            counts[name] += 1

print('\n=== surface error frequency (pred has it, gold does not) ===')
for name, _ in patterns.items():
    print(f'{counts[name]:4d}  {name}')

# --- how many declarations have AT LEAST ONE surface error ---
any_surface = 0
for p, g in zip(preds, golds):
    if any(has(p, g, pat) for pat in patterns.values()):
        any_surface += 1
print(f'\ndeclarations with >=1 surface error: {any_surface} ({any_surface/N:.1%})')
