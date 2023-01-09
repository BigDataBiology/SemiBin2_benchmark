import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib_venn import venn3_unweighted

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

project_id = ['PRJCA007414', 'PRJNA595610', 'PRJEB48021']


def get_result(result_path):
    result = pd.read_csv(result_path, index_col=0)
    near_complete = result[(result['Completeness'].astype(float) > float(90)) & (result['Contamination'].astype(float) < float(0.05 * 100)) & (result['pass.GUNC'] == True)]
    high_qulaity = result[(result['Completeness'].astype(float) > float(90)) & (result['Contamination'].astype(float) < float(0.05 * 100)) & (result['pass.GUNC'] == True) & (result['tRNA'] >= 18) & (result['23S'] >= 1) & (result['16S'] >= 1) & (result['5S'] >= 1)]
    return near_complete.shape[0], high_qulaity.shape[0]

def plot():
    for project in ['PRJCA007414','PRJNA595610', 'PRJEB48021']:

        near_list = []
        high_list = []

        if project == 'PRJCA007414':
            method_list = ['LRBinner', 'Metabat2', 'VAMB', 'SemiBin', 'GraphMB', 'Metadecoder', 'SemiBin2_train']
        else:
            method_list = ['LRBinner', 'Metabat2', 'VAMB', 'SemiBin', 'GraphMB', 'Metadecoder', 'SemiBin2']
        run_list = os.listdir(f'data/real_long/{project}')
        for method in method_list:
            num_near = 0
            num_high = 0
            for run in run_list:
                result_file = f'data/real_long/{project}/{run}/{method}/result.csv'
                near_complete, high_quality = get_result(result_file)
                num_near += near_complete
                num_high += high_quality
            near_list.append(num_near)
            high_list.append(num_high)
        subset = pd.DataFrame(np.array([near_list,high_list]),columns = method_list, index=['Near-complete','High-quality'])
        print(subset)

        ax = subset.plot(kind='bar',color = ['#969696', '#dd3497','#c7eae5', '#7570b3', '#e6ab02', '#8c510a', '#1b9e77'], figsize=(6,3))
        ax.set_xticklabels(labels=['Near-complete','High-quality'], fontsize=15,color = 'black',rotation = 360)
        ax.set_ylabel('Number of bins', fontsize=15,color = 'black')
        ax.set_title(f'{project}', fontsize=20, alpha=1.0,color = 'black')
        plt.savefig(f'{project}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_checkm():
    for project in ['PRJCA007414','PRJNA595610', 'PRJEB48021']:

        near_list = []
        high_list = []

        if project == 'PRJCA007414':
            method_list = ['LRBinner', 'Metabat2', 'VAMB', 'SemiBin', 'GraphMB', 'Metadecoder', 'SemiBin2_train']
        else:
            method_list = ['LRBinner', 'Metabat2', 'VAMB', 'SemiBin', 'GraphMB', 'Metadecoder', 'SemiBin2']
        run_list = os.listdir(f'data/real_long/{project}')
        for method in method_list:
            num_near = 0
            num_high = 0
            for run in run_list:
                result_file = f'data/real_long/{project}/{run}/{method}/result_checkm.csv'
                near_complete, high_quality = get_result(result_file)
                num_near += near_complete
                num_high += high_quality
            near_list.append(num_near)
            high_list.append(num_high)
        subset = pd.DataFrame(np.array([near_list,high_list]),columns = method_list, index=['Near-complete','High-quality'])
        print(subset)

        ax = subset.plot(kind='bar',color = ['#969696', '#dd3497','#c7eae5', '#7570b3', '#e6ab02', '#8c510a', '#1b9e77'], figsize=(6,3))
        ax.set_xticklabels(labels=['Near-complete','High-quality'], fontsize=15,color = 'black',rotation = 360)
        ax.set_ylabel('Number of bins', fontsize=15,color = 'black')
        ax.set_title(f'{project}', fontsize=20, alpha=1.0,color = 'black')
        plt.savefig(f'{project}_checkm.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_pretrain():
    for project in ['PRJCA007414','PRJNA595610', 'PRJEB48021']:

        near_list = []
        high_list = []

        if project == 'PRJCA007414':
            method_list = ['SemiBin2', 'SemiBin2_train']
        else:
            method_list = ['SemiBin2_pretrain', 'SemiBin2']
        run_list = os.listdir(f'data/real_long/{project}')
        for method in method_list:
            num_near = 0
            num_high = 0
            for run in run_list:
                result_file = f'data/real_long/{project}/{run}/{method}/result.csv'
                near_complete, high_quality = get_result(result_file)
                num_near += near_complete
                num_high += high_quality
            near_list.append(num_near)
            high_list.append(num_high)
        subset = pd.DataFrame(np.array([near_list,high_list]),columns = method_list, index=['Near-complete','High-quality'])
        print(subset)

        ax = subset.plot(kind='bar',color = ['#dd3497', '#1b9e77'], figsize=(6,3))
        ax.set_xticklabels(labels=['Near-complete','High-quality'], fontsize=15,color = 'black',rotation = 360)
        ax.set_ylabel('Number of bins', fontsize=15,color = 'black')
        ax.set_title(f'{project}', fontsize=20, alpha=1.0,color = 'black')
        plt.savefig(f'{project}_pretrain.pdf', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    plot_pretrain()
    plot_checkm()
    plot()