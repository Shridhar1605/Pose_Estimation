import json
with open('model-training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('outputs.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb['cells']):
        if 'outputs' in cell:
            f.write(f"\n--- CELL {i} OUTPUT ---\n")
            for o in cell['outputs']:
                if 'text' in o:
                    f.write("".join(o['text']))
                elif 'data' in o and 'text/plain' in o['data']:
                    f.write("".join(o['data']['text/plain']))
