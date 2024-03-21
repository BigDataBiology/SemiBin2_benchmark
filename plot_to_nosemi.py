import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

skin_index = [1, 13, 14, 15, 16, 17, 18, 19, 20, 28]
oral_index = [6,7,8,13,14,15,16,17,18,19]
airways_index = [4,7,8,9,10,11,12,23,26,27]
gastrointestinal_index = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12]
urogenital_index= [0, 2, 3, 5, 6, 21, 22, 24, 25]

def plot_short_reads():
    strain_improvement = []
    species_improvement = []
    genus_improvement = []

    for env in ['Airways','Gastrointestinal','Oral','Skin','Urogenital']:
        print(env)
        data_index = None
        if env == 'Skin':
            data_index = skin_index
        if env == 'Oral':
            data_index = oral_index
        if env == 'Airways':
            data_index = airways_index
        if env == 'Gastrointestinal':
            data_index = gastrointestinal_index
        if env == 'Urogenital':
            data_index = urogenital_index

        genome_path = os.path.join('data/short_reads/{0}/amber_{1}'.format(env, data_index[0]), 'genome')
        method_list = {}
        species_list = {}
        genus_list = {}

        for root, dirs, files in os.walk(genome_path, topdown=False):
            for name in dirs:
                method_list[name] = []
                species_list[name] = []
                genus_list[name] = []

        for temp in data_index:
            taxi = pd.read_csv('data/short_reads/{0}/taxonomic_profile.txt'.format(env), sep='\t', skiprows=3, dtype={'@@TAXID': str, 'TAXPATH': str})
            taxi_genus = taxi[taxi['RANK'] == 'genus']['@@TAXID'].values.tolist()
            taxi_species = taxi[taxi['RANK'] == 'species'][
                '@@TAXID'].values.tolist()
            method_path_list = []
            genome_path = os.path.join('data/short_reads/{0}/amber_{1}'.format(env, temp),'genome')

            for root, dirs, files in os.walk(genome_path, topdown=False):
                for name in dirs:
                    method_path_list.append(os.path.join(root, name))

            for method_path in method_path_list:
                metric = pd.read_csv(os.path.join(method_path, 'metrics_per_bin.tsv'), sep='\t')
                com_90_pur_95 = metric[
                    (metric['Completeness (bp)'].astype(float) > float(0.9)) & (
                            metric['Purity (bp)'].astype(float) >= float(0.95))]
                strain_list = com_90_pur_95['Most abundant genome'].values.tolist()
                method_list[method_path.split('/')[-1]].extend(strain_list)

                for temp_strain in strain_list:
                    if temp_strain in taxi['_CAMI_GENOMEID'].values.tolist():
                        taxi_split = taxi[taxi['_CAMI_GENOMEID'] == temp_strain]['TAXPATH'].values[0].split('|')
                        if taxi_split[-2] in taxi_species:
                            species_list[method_path.split('/')[-1]].append(taxi_split[-2])
                        if taxi_split[-3] in taxi_genus:
                            genus_list[method_path.split('/')[-1]].append(taxi_split[-3])

        result = {'NoSemi': {'strain': list(set(method_list['NoSemi'])),
                             'species': list(set(species_list['NoSemi'])),
                             'genus': list(set(genus_list['NoSemi']))},
                  'SemiBin2': {'strain': list(set(method_list['SemiBin2'])),
                               'species': list(set(species_list['SemiBin2'])),
                               'genus': list(set(genus_list['SemiBin2']))}}

        strain_nosemi = len(result['NoSemi']['strain'])
        species_nosemi = len(result['NoSemi']['species'])
        genus_nosemi = len(result['NoSemi']['genus'])

        strain_SemiBin2 = len(result['SemiBin2']['strain'])
        species_SemiBin2 = len(result['SemiBin2']['species'])
        genus_SemiBin2 = len(result['SemiBin2']['genus'])

        print(strain_SemiBin2, species_SemiBin2, genus_SemiBin2)
        print(strain_nosemi, species_nosemi, genus_nosemi)

        print('strain_improvement(NoSemi): ', (strain_SemiBin2 - strain_nosemi) / strain_nosemi)
        strain_improvement.append((strain_SemiBin2 - strain_nosemi) / strain_nosemi)
        print('species_improvement(NoSemi): ', (species_SemiBin2 - species_nosemi) / species_nosemi)
        species_improvement.append((species_SemiBin2 - species_nosemi)/ species_nosemi)
        print('genus_improvement(NoSemi): ', (genus_SemiBin2 - genus_nosemi) / genus_nosemi)
        genus_improvement.append((genus_SemiBin2 - genus_nosemi)/ genus_nosemi)

        line_width = 1

        plt.figure(figsize=(4, 4))
        plt.plot(['genus', 'species', 'strain'],
                 [genus_nosemi, species_nosemi,
                  strain_nosemi], label='NoSemi', color='#ec7014',
                 linewidth=line_width, marker='o', )

        plt.plot(['genus', 'species', 'strain'],
                 [genus_SemiBin2, species_SemiBin2,
                  strain_SemiBin2], label='SemiBin2',
                 color='#1b9e77', linewidth=line_width, marker='o', )
        plt.legend()
        plt.title(f"{env}", fontsize=20, alpha=1.0, color='black')
        plt.savefig(f'cami2_short_reads_{env}_nosemi.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    print('Average strain improvement:', np.mean(strain_improvement))
    print('Average species improvement:', np.mean(species_improvement))
    print('Average genus improvement:', np.mean(genus_improvement))

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
        data = get_number_of_genomes_per_completeness(f'data/long_reads/amber_{env}', return_pandas=True)

        subset = data.loc[[
            'NoSemi',
            'SemiBin2']]
        subset = subset[[90, 80, 70, 60]]
        print(subset)
        high_quality_list = subset[90].sort_values().values
        print('Improvement of best binner over second best: {:.2%}'.format((high_quality_list[-1] - high_quality_list[-2]) / high_quality_list[-2]))
        improvements.append((high_quality_list[-1] - high_quality_list[-2])  / high_quality_list[-2] )
        ax = subset.plot(kind="barh", stacked=True, legend=False,
                         color=color_map)

        ax.legend(['>90', '>80', '>70', '>60'],
                  loc='lower right', fontsize=10, title='completeness')
        # ax.set_xticks(ticks=y_label)
        # ax.set_xticklabels(labels= fontsize=15, color='black')
        plt.xticks(fontsize=15, color='black')
        ax.set_yticklabels(labels=subset.index, fontsize=15, color='black')
        ax.set_xlabel('Bins(< 5% contamination)', fontsize=15, color='black')
        ax.set_title(f'{env}', fontsize=15, alpha=1.0)

        plt.tight_layout()
        plt.savefig(f'cami2_long_reads_{env}_nosemi.pdf', dpi=300)
        plt.close()
        plt.show()
    print('Average improvements:', np.mean(improvements))

if __name__ == '__main__':
    # plot_short_reads()
    plot_long_reads()