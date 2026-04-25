import json
import re

path = '/root/.openclaw/workspace/projects/maneuvers/examples/notebooks/detection_classification_demo.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

venv_python = '/root/.openclaw/workspace/projects/maneuvers/.venv/bin/python'

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace ['python', ... with ['/path/to/venv/bin/python', ...
            line = re.sub(r"subprocess\.run\(\s*\[\s*['\"]python['\"],", f"subprocess.run(['{venv_python}',", line)
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
