import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def get_number_of_genomes_per_completeness(amber_path, return_pandas=False):
    genome_path = os.path.join(amber_path, 'genome')
    table = {}
    for root, dirs, files in os.walk(genome_path, topdown=False):
        for name in dirs:
            method_path = os.path.join(root, name)
            metric = pd.read_csv(os.path.join(method_path, 'metrics_per_bin.tsv'), sep='\t')
            metric = metric.query('`Purity (bp)` >= 0.95')
            com_90_pur_95 = metric.eval('`Completeness (bp)` > 0.9').sum()
            com_80_pur_95 = metric.eval('`Completeness (bp)` > 0.8').sum() - (
                        com_90_pur_95)
            com_70_pur_95 = metric.eval('`Completeness (bp)` > 0.7').sum() - (
                        com_90_pur_95 + com_80_pur_95)
            com_60_pur_95 = metric.eval('`Completeness (bp)` > 0.6').sum() - (
                        com_90_pur_95 + com_80_pur_95 + com_70_pur_95)
            com_50_pur_95 = metric.eval('`Completeness (bp)` > 0.5').sum() - (
                        com_90_pur_95 + com_80_pur_95 + com_70_pur_95 + com_60_pur_95)
            table[method_path.split('/')[-1]]= [com_90_pur_95, com_80_pur_95, com_70_pur_95, com_60_pur_95, com_50_pur_95]
    if return_pandas:
        return pd.DataFrame(table, index=[90,80,70,60,50]).T
    return table

color_map = ['#01665e', '#5da8a1', '#80cdc1', '#c7eae5']
def plot_long_reads():
    improvements = []
    for env in ['Airways','Gastrointestinal','Oral','Skin','Urogenital']:
        print(env)
        data = get_number_of_genomes_per_completeness(f'data/long_reads_ensemble/amber_{env}', return_pandas=True)
        columns = [
            '0.01',
            '0.05',
            '0.1',
            '0.15',
            '0.2',
            '0.25',
            '0.3',
            '0.35',
            '0.4',
            '0.45',
            '0.5',
            '0.55',
            'SemiBin2'
        ]
        subset = data.loc[columns]
        subset = subset[[90]].values.tolist()
        subset = [value[0] for value in subset]
        print('Improvements:', (subset[-1] - subset[0]) / subset[0])
        improvements.append(subset)
    df = pd.DataFrame(improvements, index=['Airways','Gastrointestinal','Oral','Skin','Urogenital'], columns=[columns])
    print(df)
    fig, ax = plt.subplots(figsize = (8,3))
    sns.heatmap(df, cmap='YlOrBr', fmt='.20g', annot=True)
    ax.set_xticklabels(columns, rotation=50)
    ax.set_xlabel('Different parameters', fontsize=15, color='black')
    ax.set_ylabel('Environment', fontsize=15, color='black')
    fig.tight_layout()
    plt.savefig('ensemble_comparison.pdf', dpi=300,  bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_long_reads()