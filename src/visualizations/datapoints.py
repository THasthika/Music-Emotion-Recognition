import matplotlib.pyplot as plt
import numpy as np
import mpld3
from os import path
import pandas as pd

BASE_DIR = "datasets/PMEmo2019"
ANNOTATION_DIR = path.join(BASE_DIR, "annotations")

std_csv_path = path.join(ANNOTATION_DIR, "static_annotations_std.csv")
mean_csv_path = path.join(ANNOTATION_DIR, "static_annotations.csv")

std_df = pd.read_csv(std_csv_path)
mean_df = pd.read_csv(mean_csv_path)

all_df = mean_df.merge(std_df, how="inner", on=["musicId"])

print(all_df.describe())

# cols = all_df.columns.tolist()
# cols = cols[0:2] + cols[3:4] + cols[2:3] + cols[4:5]
# all_df = all_df[cols]

# all_df[["musicId"]] = all_df[["musicId"]].astype(str)

# fig, ax = plt.subplots()
# scatter = ax.scatter(all_df.iloc[:, 3], all_df.iloc[:, 1], cmap=plt.cm.jet)

# ax.set_title("OK!")

# ax.set_ylabel("Arousal")
# ax.set_xlabel("Valence")

# tooltip = mpld3.plugins.PointLabelTooltip(scatter)
# mpld3.plugins.connect(fig, tooltip)

# mpld3.show()

# fig, ax = plt.subplots()
# N = 100

# scatter = ax.scatter(np.random.normal(size=N),
#                      np.random.normal(size=N),
#                      c=np.random.random(size=N),
#                      s=1000 * np.random.random(size=N),
#                      alpha=0.3,
#                      cmap=plt.cm.jet)
# ax.grid(color='white', linestyle='solid')

# ax.set_title("Scatter Plot (with tooltips!)", size=20)

# labels = ['point {0}'.format(i + 1) for i in range(N)]

# tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
# mpld3.plugins.connect(fig, tooltip)

# mpld3.show()