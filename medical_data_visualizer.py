import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype('int')

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype('int')
df['gluc'] = (df['gluc'] > 1).astype('int')

# 4
def draw_cat_plot(): 
    # 5, 6
    feats = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    df_cat = pd.melt(df, id_vars='cardio', value_vars=feats)
    
    # 7
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat = df_cat.rename(columns={0: 'total'})

    # 8
    chart = sns.catplot(data=df_cat, kind="bar", x="variable", y="total", hue="value", col="cardio")
    fig = chart.fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df.height.quantile(0.025) <= df.height) & 
        (df.height <= df.height.quantile(0.975)) &
        (df.weight.quantile(0.025) <= df.weight) &
        (df.weight <= df.weight.quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(16,9))

    # 15
    sns.heatmap(corr, mask=mask, linewidths=0.5, square=True, annot=True, fmt='0.1f')

    # 16
    fig.savefig('heatmap.png')
    return fig

draw_heat_map()