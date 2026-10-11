"""Shared house style for the 261011 revision figures: serif, no in-chart title, minimalist; captions live in main.tex."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"J15": "#93B2DD", "J3": "#4A78BE", "J7": "#0F3F7A", "JL": "#E0908E", "JL8": "#A8322F",
       "JO1": "#8DCBD2", "JO7": "#2A7680"}
LAB = {"J15": "Q1.5", "J3": "Q3", "J7": "Q7", "JL": "L3", "JL8": "L8", "JO1": "O1", "JO7": "O7"}
ORDER = ["J15", "J3", "J7", "JL", "JL8", "JO1", "JO7"]
