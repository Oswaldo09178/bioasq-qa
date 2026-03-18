import matplotlib.pyplot as plt
import numpy as np

question_types = ['yesno', 'factoid', 'list', 'summary', 'overall']

bm25 =    [0.4041, 0.3279, 0.2438, 0.2434, 0.3117]
hybrid =  [0.5585, 0.5077, 0.4246, 0.4692, 0.4958]
reranked =[0.6731, 0.6849, 0.6081, 0.7043, 0.6712]

x = np.arange(len(question_types))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, bm25,    width, label='BM25 Only',              color='#d9534f')
ax.bar(x,         hybrid,  width, label='Hybrid (BM25 + BGE-M3)', color='#f0ad4e')
ax.bar(x + width, reranked,width, label='Hybrid + Cross-Encoder',  color='#5cb85c')

ax.set_xlabel('Question Type')
ax.set_ylabel('MAP@10')
ax.set_title('MedConvoQA Retrieval Performance by Question Type')
ax.set_xticks(x)
ax.set_xticklabels(question_types)
ax.set_ylim(0, 0.85)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bars in ax.containers:
    ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

plt.tight_layout()
plt.savefig('output/retrieval_comparison.png', dpi=150)
print("[INFO] Chart saved!")
