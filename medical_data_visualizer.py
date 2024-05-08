import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
height_mtrs = df['height'] / 100
df['calc_BMI'] = df['weight'] / (height_mtrs ** 2)
df['overweight'] = None
df.loc[df['calc_BMI'] > 25, 'overweight'] = 1
df.loc[df['calc_BMI'] <= 25, 'overweight'] = 0
df = df.drop('calc_BMI', axis = 'columns')  # Una vez que usé la columna 'calc_BMI' para calcular el 'overweight' puedo descatarla

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.loc[df['cholesterol'] <= 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] <= 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], id_vars = 'cardio')


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable'])['value'].value_counts()
    df_cat = df_cat.to_frame()
    df_cat = df_cat.rename(columns={'count': 'total'})
    df_cat = df_cat.reset_index()
    

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(data = df_cat, kind = 'bar', x = 'variable', y = 'total', col = 'cardio', order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], hue = 'value')


    # Get the figure for the output
    fig = fig.fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))


    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(df_heat.corr(), mask=mask,
        annot=True, center=0,
        linewidths=0.5, square=True,
        vmin=-0.15, vmax=0.3, fmt='0.1f', annot_kws={"size":7}, cbar_kws={"shrink": .7})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
