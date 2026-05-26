import os

enz = 'Mpro2'
seqs = [
    'AVLQSGFR', 'VILQSGFR', 'VILQTGFR', 'VILQSPFR',
    'VILHSGFR', 'VIMQSGFR', 'VPLQSGFR', 'NILQSGFR',
]

path = os.path.join('sequences', 'predict')
savePath = os.path.join(path, f'{enz}_subsPred_{len(seqs[0])}AA.csv')
print(f'Saving sequences at:\n  {savePath.replace("../", "")}')
if not os.path.exists(path):
    os.makedirs(path)

with open(savePath, 'w') as f:
    f.write('sequence\n')
    for seq in seqs:
        f.write(seq + '\n')