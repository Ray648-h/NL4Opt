import json, random, shutil

SRC = '/home/ray/projects/nl4opt/nl4opt-competition/generation_data'
DST = '/home/ray/projects/nl4opt/nl4opt-subtask2-baseline/data'

def ids_of(lines):
    ids = set()
    for line in lines:
        ids.update(json.loads(line).keys())
    return ids

with open(f'{SRC}/train.jsonl') as f:
    train_lines = f.readlines()
with open(f'{SRC}/dev.jsonl') as f:
    dev_lines = f.readlines()

random.seed(42)
random.shuffle(train_lines)
train200 = train_lines[:200]

train_ids = ids_of(train200)
dev_ids = ids_of(dev_lines)
overlap = train_ids & dev_ids

print(f'full train: {len(train_lines)}  full dev: {len(dev_lines)}')
print(f'-> train200: {len(train200)}  -> dev (kept whole): {len(dev_lines)}')
print(f'doc_id overlap between train200 and dev: {len(overlap)}')

with open(f'{DST}/train.jsonl', 'w') as f:
    f.writelines(train200)
shutil.copy(f'{SRC}/dev.jsonl', f'{DST}/dev.jsonl')
print('written data/train.jsonl (200) and data/dev.jsonl (99)')
