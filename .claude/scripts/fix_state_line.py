import json

NB_PATH = 'E:/UIT/study/claude/learn-claude/papers/13 - GrapeLeafNet/ouputs/grapeleafnet.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][10]['source'])

# Remove the stale duplicate save line
src = src.replace(
    "            torch.save(model.state_dict(), checkpoint_path)\n"
    "            torch.save(_state, checkpoint_path)\n",
    "            torch.save(model.state_dict(), checkpoint_path)\n"
)

nb['cells'][10]['source'] = src
nb['cells'][10]['outputs'] = []

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed.")
