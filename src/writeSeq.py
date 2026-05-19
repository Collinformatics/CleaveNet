import os

enz = 'Mpro2'
seqs = [
    'AVLQSGFR', 'VILQSGFR', 'VILQTGFR', 'VILQSPFR',
    'VILHSGFR', 'VIMQSGFR', 'VPLQSGFR', 'NILQSGFR',
]

saveDir  = os.path.join('../splits', 'predict')
savePath = os.path.join(saveDir, f'{enz}_subsPred.csv')
print(f'Saving sequences at:\n  {savePath.replace("../", "")}')
if not os.path.exists(savePath):
    os.makedirs(saveDir)

with open(savePath, 'w') as f:
    f.write('sequence\n')
    for seq in seqs:
        f.write(seq + '\n')