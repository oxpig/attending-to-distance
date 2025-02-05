import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('aa_recoveries.csv')

x = np.arange(len(df['aa']))
width = 0.4

# plt.bar(df['aa'], df['coords_rate'], label='Coords', edgecolor='black', hatch="//", color='orange')
# plt.bar(df['aa'], df['no_coords_rate'], label='No coords', edgecolor='black', hatch = "//\\\\", color='orange')

plt.bar(x, df['no_coords_rate'], width, label='No coords')
plt.bar(x+width, df['coords_rate'], width, label='Coords')
plt.legend()
plt.ylabel('Recovery rate', fontsize=16)
plt.xlabel('Amino acid', fontsize=16)
plt.xticks(x+width/2, df['aa'], fontsize=16)
# plt.show()
plt.tight_layout()
plt.savefig('aa_recoveries.pdf')
