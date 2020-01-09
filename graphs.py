import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('stats.csv')

pubs = list(df['publication'])[:15]
twc = list(df['pub_word_count'])[:15]

fig, ax = plt.subplots()
ax.bar(pubs,twc)

for label in ax.get_xticklabels():
    label.set_ha('right')
    label.set_rotation(90)

plt.tight_layout()
plt.xlabel('Publication')
plt.ylabel('Word Count')
plt.title('Total Word Count by Publication')
plt.show()


