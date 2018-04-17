import pandas as pd
import matplotlib.pyplot as plt

# Plot the evolution of the IoU with the number of interactions
df = pd.read_csv('results/result_default_settings.csv', index_col=0)
df.groupby('interaction').jaccard.mean().plot()
plt.ylabel('IoU')
plt.xlabel('Number of Interactions')
plt.show()
