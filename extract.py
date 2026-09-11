import json

with open('model-training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook_code.py', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            f.write(f"# CELL {i}\n")
            f.write("".join(cell['source']) + "\n\n")
