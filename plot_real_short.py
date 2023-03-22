import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3_unweighted

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

human_list = ['CCMD41521570ST', 'CCMD75147712ST', 'CCMD18579000ST', 'CCMD53508245ST', 'CCMD19168690ST', 'CCMD52117727ST', 'CCMD42956136ST', 'CCMD79349503ST', 'CCMD89306485ST', 'CCMD76409700ST', 'CCMD31134579ST', 'CCMD71242853ST', 'CCMD89107682ST', 'CCMD76222476ST', 'CCMD10032470ST', 'CCMD17410933ST', 'CCMD38158721ST', 'CCMD35081859ST', 'CCMD54057834ST', 'CCMD28738636ST', 'CCMD98702133ST', 'CCMD30626189ST', 'CCMD32965613ST', 'CCMD53522274ST', 'CCMD37575804ST', 'CCMD68973846ST', 'CCMD25475945ST', 'CCMD65406197ST', 'CCMD21703880ST', 'CCMD50300306ST', 'CCMD51228890ST', 'CCMD59540613ST', 'CCMD49942357ST', 'CCMD95431029ST', 'CCMD41202658ST', 'CCMD15562448ST', 'CCMD21593359ST', 'CCMD92404903ST', 'CCMD50538120ST', 'CCMD49461418ST', 'CCMD72690923ST', 'CCMD85481373ST', 'CCMD39882286ST', 'CCMD18829815ST', 'CCMD51154251ST', 'CCMD85661207ST', 'CCMD71915439ST', 'CCMD39157124ST', 'CCMD22852639ST', 'CCMD35801800ST', 'CCMD27463710ST', 'CCMD59583015ST', 'CCMD89967135ST', 'CCMD52145360ST', 'CCMD95676152ST', 'CCMD45004878ST', 'CCMD67373733ST', 'CCMD99929634ST', 'CCMD89643949ST', 'CCMD26625622ST', 'CCMD23541216ST', 'CCMD31009081ST', 'CCMD99440714ST', 'CCMD66848156ST', 'CCMD65222621ST', 'CCMD98531134ST', 'CCMD45812507ST', 'CCMD46727384ST', 'CCMD73128545ST', 'CCMD30627121ST', 'CCMD50529145ST', 'CCMD98198513ST', 'CCMD93755960ST', 'CCMD35633353ST', 'CCMD56948710ST', 'CCMD27867141ST', 'CCMD32288175ST', 'CCMD29706695ST', 'CCMD72666896ST', 'CCMD10191450ST', 'CCMD49025643ST', 'CCMD74592084ST']

dog_list = ['SAMN06172456', 'SAMN06172425', 'SAMN06172487', 'SAMN06172450', 'SAMN06172459', 'SAMN06172479', 'SAMN06172435', 'SAMN06172414', 'SAMN06172409', 'SAMEA103957796', 'SAMN06172442', 'SAMN06172500', 'SAMN06172437', 'SAMN06172413', 'SAMN06172514', 'SAMN06172403', 'SAMN06172471', 'SAMN06172490', 'SAMN06172448', 'SAMN06172504', 'SAMN06172457', 'SAMN06172441', 'SAMN06172422', 'SAMN06172408', 'SAMN06172429', 'SAMN06172420', 'SAMN06172503', 'SAMN06172410', 'SAMN06172458', 'SAMN06172493', 'SAMEA103957794', 'SAMN06172402', 'SAMN06172515', 'SAMN06172462', 'SAMN06172421', 'SAMN06172411', 'SAMN06172511', 'SAMN06172516', 'SAMN06172465', 'SAMN06172419', 'SAMN06172517', 'SAMN06172510', 'SAMN06172418', 'SAMN06172424', 'SAMN06172427', 'SAMN06172453', 'SAMN06172491', 'SAMN06172496', 'SAMN06172513', 'SAMN06172461', 'SAMN06172449', 'SAMN06172426', 'SAMN06172452', 'SAMN06172522', 'SAMN06172400', 'SAMN06172405', 'SAMN06172521', 'SAMN06172407', 'SAMN06172455', 'SAMN06172446', 'SAMN06172467', 'SAMN06172499', 'SAMN06172474', 'SAMN06172412', 'SAMN06172468', 'SAMN06172478', 'SAMN06172423', 'SAMN06172447', 'SAMN06172415', 'SAMN06172523', 'SAMN06172417', 'SAMN06172497', 'SAMN06172498', 'SAMN06172489', 'SAMN06172436', 'SAMN06172432', 'SAMN06172406', 'SAMN06172488', 'SAMN06172502', 'SAMN06172401', 'SAMN06172434', 'SAMN06172416', 'SAMN06172445', 'SAMN06172431', 'SAMN06172438', 'SAMN06172473', 'SAMN06172486', 'SAMN06172472', 'SAMN06172428', 'SAMEA103957793', 'SAMEA103957795', 'SAMN06172443', 'SAMN06172475', 'SAMN06172520', 'SAMN06172495', 'SAMN06172440', 'SAMN06172430', 'SAMN06172481', 'SAMN06172524', 'SAMN06172519', 'SAMN06172454', 'SAMN06172404', 'SAMN06172460', 'SAMN06172433', 'SAMN06172469', 'SAMN06172451', 'SAMN06172476', 'SAMN06172492', 'SAMN06172484', 'SAMN06172509', 'SAMN06172506', 'SAMN06172518', 'SAMN06172477', 'SAMN06172470', 'SAMN06172482', 'SAMN06172512', 'SAMN06172494', 'SAMN06172485', 'SAMN06172508', 'SAMN06172466', 'SAMN06172507', 'SAMN06172444', 'SAMN06172505', 'SAMN06172464', 'SAMN06172439', 'SAMN06172501', 'SAMN06172483', 'SAMN06172463', 'SAMN06172480']

ocean_list = ['TARA_041_SRF_0.1-0.22', 'TARA_038_SRF_0.22-1.6', 'TARA_076_SRF_0.22-3', 'TARA_023_SRF_0.22-1.6', 'TARA_042_SRF_0.22-1.6', 'TARA_124_SRF_0.22-3', 'TARA_124_SRF_0.22-0.45', 'TARA_066_SRF_lt-0.22', 'TARA_057_SRF_0.22-3', 'TARA_124_SRF_0.45-0.8', 'TARA_004_SRF_0.22-1.6', 'TARA_018_SRF_0.22-1.6', 'TARA_070_SRF_0.22-0.45', 'TARA_034_SRF_lt-0.22', 'TARA_064_SRF_0.22-3', 'TARA_125_SRF_0.22-0.45', 'TARA_111_SRF_0.22-3', 'TARA_122_SRF_0.22-0.45', 'TARA_145_SRF_0.22-3', 'TARA_099_SRF_0.22-3', 'TARA_038_SRF_lt-0.22', 'TARA_082_SRF_0.22-3', 'TARA_041_SRF_lt-0.22', 'TARA_146_SRF_0.22-3', 'TARA_151_SRF_0.22-3', 'TARA_123_SRF_0.22-3', 'TARA_110_SRF_0.22-3', 'TARA_150_SRF_0.22-3', 'TARA_072_SRF_lt-0.22', 'TARA_085_SRF_0.22-3', 'TARA_098_SRF_0.22-3', 'TARA_078_SRF_0.22-3', 'TARA_149_SRF_0.22-3', 'TARA_094_SRF_0.22-3', 'TARA_068_SRF_0.22-3', 'TARA_148_SRF_0.22-3', 'TARA_067_SRF_0.45-0.8', 'TARA_018_SRF_lt-0.22', 'TARA_138_SRF_0.22-3', 'TARA_093_SRF_0.22-3', 'TARA_041_SRF_0.22-1.6', 'TARA_122_SRF_0.22-3', 'TARA_078_SRF_0.45-0.8', 'TARA_070_SRF_lt-0.22', 'TARA_065_SRF_lt-0.22', 'TARA_122_SRF_0.1-0.22', 'TARA_036_SRF_0.22-1.6', 'TARA_031_SRF_0.22-1.6', 'TARA_142_SRF_0.22-3', 'TARA_124_SRF_0.1-0.22', 'TARA_036_SRF_lt-0.22', 'TARA_065_SRF_0.22-3', 'TARA_067_SRF_lt-0.22', 'TARA_112_SRF_0.22-3', 'TARA_109_SRF_0.22-3', 'TARA_068_SRF_lt-0.22', 'TARA_109_SRF_lt-0.22', 'TARA_064_SRF_lt-0.22', 'TARA_048_SRF_0.22-1.6', 'TARA_034_SRF_0.22-1.6', 'TARA_070_SRF_0.45-0.8', 'TARA_025_SRF_lt-0.22', 'TARA_133_SRF_0.22-3', 'TARA_096_SRF_0.22-3', 'TARA_038_SRF_0.1-0.22', 'TARA_007_SRF_0.22-1.6', 'TARA_048_SRF_0.1-0.22', 'TARA_140_SRF_0.22-3', 'TARA_034_SRF_0.1-0.22', 'TARA_067_SRF_0.22-3', 'TARA_125_SRF_0.45-0.8', 'TARA_030_SRF_0.22-1.6', 'TARA_031_SRF_lt-0.22', 'TARA_032_SRF_0.22-1.6', 'TARA_070_SRF_0.22-3', 'TARA_132_SRF_0.22-3', 'TARA_076_SRF_lt-0.22', 'TARA_125_SRF_0.1-0.22', 'TARA_123_SRF_0.45-0.8', 'TARA_078_SRF_lt-0.22', 'TARA_068_SRF_0.45-0.8', 'TARA_068_SRF_0.22-0.45', 'TARA_067_SRF_0.22-0.45', 'TARA_100_SRF_0.22-3', 'TARA_122_SRF_0.45-0.8', 'TARA_137_SRF_0.22-3', 'TARA_076_SRF_0.22-0.45', 'TARA_125_SRF_0.22-3', 'TARA_078_SRF_0.22-0.45', 'TARA_076_SRF_0.45-0.8', 'TARA_084_SRF_0.22-3', 'TARA_032_SRF_lt-0.22', 'TARA_025_SRF_0.22-1.6', 'TARA_062_SRF_0.22-3', 'TARA_066_SRF_0.22-3', 'TARA_036_SRF_0.1-0.22', 'TARA_056_SRF_0.22-3', 'TARA_072_SRF_0.22-3', 'TARA_128_SRF_0.22-3', 'TARA_052_SRF_0.22-1.6', 'TARA_033_SRF_0.22-1.6', 'TARA_123_SRF_0.22-0.45', 'TARA_102_SRF_0.22-3', 'TARA_065_SRF_0.1-0.22', 'TARA_009_SRF_0.22-1.6', 'TARA_141_SRF_0.22-3', 'TARA_045_SRF_0.22-1.6', 'TARA_042_SRF_lt-0.22', 'TARA_152_SRF_0.22-3']

soil_list = ['SAMN06268061', 'SAMN06268063', 'SAMN06264885', 'SAMN06267090',
                 'SAMN06267080', 'SAMN06266457', 'SAMN05421921', 'SAMN06264649',
                 'SAMN06264650', 'SAMN07631258', 'SAMN06264630', 'SAMN06267102',
                 'SAMN06267104', 'SAMN06264384', 'SAMN06266487', 'SAMN06266460',
                 'SAMN06266447', 'SAMN06264634', 'SAMN06266424', 'SAMN06268167',
                 'SAMN06266446', 'SAMN06267099', 'SAMN06266484', 'SAMN06266459',
                 'SAMN06267079', 'SAMN06267094', 'SAMN06264884', 'SAMN06266490',
                 'SAMN06266453', 'SAMN06264385', 'SAMN06264631', 'SAMN06266388',
                 'SAMN06266336', 'SAMN06266479', 'SAMN06266485', 'SAMN06267092',
                 'SAMN07631257', 'SAMN06266454', 'SAMN06266483', 'SAMN06268059',
                 'SAMN06267098', 'SAMN06268058', 'SAMN06268170', 'SAMN06266423',
                 'SAMN06264948', 'SAMN06267083', 'SAMN06264648', 'SAMN07631255',
                 'SAMN06266478', 'SAMN06267085', 'SAMN06266448', 'SAMN06267101',
                 'SAMN06268168', 'SAMN06267097', 'SAMN06266475', 'SAMN06266450',
                 'SAMN06264881', 'SAMN06264635', 'SAMN06267088', 'SAMN06266458',
                 'SAMN06267095', 'SAMN06264383', 'SAMN06266461', 'SAMN06266449',
                 'SAMN06267096', 'SAMN06267087', 'SAMN06267103', 'SAMN06266486',
                 'SAMN06267084', 'SAMN06264882', 'SAMN06266387', 'SAMN06266473',
                 'SAMN05421920', 'SAMN06266491', 'SAMN05421922', 'SAMN06267086',
                 'SAMN06264947', 'SAMN06268062', 'SAMN07631256', 'SAMN06267100',
                 'SAMN06267091', 'SAMN06268166', 'SAMN06264632', 'SAMN06264883',
                 'SAMN06266481', 'SAMN06266482', 'SAMN06267089', 'SAMN06267093',
                 'SAMN06264633', 'SAMN06266456', 'SAMN06267081', 'SAMN06266474',
                 'SAMN06267105', 'SAMN05421524', 'SAMN05421649', 'SAMN06268169',
                 'SAMN06267082', 'SAMN06268060', 'SAMN06266477', 'SAMN06266489',
                 'SAMN06266455']

contamination = 0.05

def get_result(dataset='dog_gut', method='SemiBin', checkm_only = False):
    """
    dataset: dog, human, gut
    method: Maxbin2, Metabat2, VAMB, S3N2Bin
    binning_mode: single_sample, multi_sample
    checkm_only: if just using checkm or using checkm and GUNC
    """
    if dataset == 'dog_gut':
        sample_list = dog_list
    elif dataset == 'soil':
        sample_list = soil_list
    elif dataset == 'human_gut':
        sample_list = human_list
    elif dataset == 'ocean':
        sample_list = ocean_list
    else:
        raise KeyError(f"Unknown dataset {dataset}")

    result = {}
    if method == 'VAMB':
        result = {'high quality': []}
        binning_result = pd.read_csv('data/real_short/{0}/VAMB_multi.csv'.format(dataset),index_col=0)
        if not checkm_only:
            high_quality = binning_result[(binning_result['Completeness'].astype(float) > float(90)) & (
                    binning_result['Contamination'].astype(float) < float(contamination * 100)) & (binning_result['pass.GUNC'] == True)]
        else:
            high_quality = binning_result[(binning_result['Completeness'].astype(float) > float(90)) & (
                    binning_result['Contamination'].astype(float) < float(contamination * 100))]
        high_quality = high_quality.index.tolist()
        result['high quality'].extend(high_quality)
        return result

    else:
        for sample in sample_list:
            result[sample] = {'high quality':[]}
            binning_result = pd.read_csv(f'data/real_short/{dataset}/{sample}/{method}/result.csv', index_col=0)
            if not checkm_only:
                high_quality = binning_result[(binning_result['Completeness'].astype(float) > float(90)) & (
                        binning_result['Contamination'].astype(float) < float(contamination * 100)) & (
                                                          binning_result['pass.GUNC'] == True)]
            else:
                high_quality = binning_result[(binning_result['Completeness'].astype(float) > float(90)) & (
                        binning_result['Contamination'].astype(float) < float(contamination * 100))]
            high_quality = high_quality.index.tolist()
            result[sample]['high quality'].extend(high_quality)

        return result


def get_results_table(dataset, method, checkm_only=False):
    r = get_result(dataset,
            method,
            checkm_only)
    if method == 'VAMB':
        from collections import Counter
        hq = r['high quality']
        r = pd.Series(Counter(b.rsplit('C', 1)[0] for b in hq)).reset_index().rename(columns={'index':'sample', 0: 'nr_hq'})
    else:
        r = pd.DataFrame([(k,len(v['high quality'])) for k,v in r.items()], columns=['sample', 'nr_hq'])
    r['dataset'] = dataset
    r['method'] = method
    r['checkm_only'] = checkm_only
    return r


from scipy.stats import wilcoxon
def plot_bar_per_sample_com(dataset, diff_label = None, num_label = None):
    num_multi = pd.concat([
            get_results_table(dataset=dataset, method='VAMB'),
            get_results_table(dataset=dataset, method='SemiBin'),
            get_results_table(dataset=dataset, method='SemiBin2'),
        ])

    counts_multi = pd.pivot(num_multi[['sample', 'nr_hq', 'method']], values=['nr_hq'], index='sample', columns='method')
    counts_multi  = counts_multi.T.xs('nr_hq').T
    counts_multi = counts_multi.fillna(0)

    counts = counts_multi
    VAMB_multi = counts['VAMB'].values
    SemiBin_multi = counts['SemiBin'].values
    SemiBin2_multi = counts['SemiBin2'].values
    print(len(VAMB_multi))
    print('SemiBin2_multi compared to VAMB_multi: {0}({1}) improvement'.format(np.sum(SemiBin2_multi) - np.sum(VAMB_multi), (np.sum(SemiBin2_multi) - np.sum(VAMB_multi))/ np.sum(VAMB_multi)))
    print('Compared to VAMB_multi wilcoxon:', wilcoxon(VAMB_multi, SemiBin2_multi))

    print('SemiBin2_multi compared to SemiBin_multi: {0}({1}) improvement'.format(np.sum(SemiBin2_multi) - np.sum(SemiBin_multi), (np.sum(SemiBin2_multi) - np.sum(SemiBin_multi))/ np.sum(SemiBin_multi)))
    print('Compared to VAMB_multi wilcoxon:', wilcoxon(SemiBin_multi, SemiBin2_multi))


    diff_multi = counts_multi.SemiBin2 - counts_multi.T
    diff = diff_multi

    fig,axes = plt.subplots(1, 2, sharex='col',figsize = (9,1.5))

    val = diff.T
    val = val[['VAMB','SemiBin','SemiBin2']]
    ax = sns.swarmplot(y='method', x='value', data=pd.melt(val), ax=axes[1], size=2, palette=['#ec7014','#7570b3','#1b9e77'])
    if diff_label is not None:
        ax.set_xticks(ticks=diff_label)
        ax.set_xticklabels(labels=diff_label)

    v = pd.DataFrame({'total': counts.sum()})
    v = v.reindex(index = ['VAMB','SemiBin','SemiBin2'])
    print(v)

    ax = sns.barplot(data=v.reset_index(), x='total', y='method', ax=axes[0], palette=['#ec7014','#7570b3','#1b9e77'])
    if num_label is not None:
        ax.set_xticks(ticks=num_label)
        ax.set_xticklabels(labels=num_label)

    ax.set_yticklabels(labels=[])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine()
    fig.tight_layout()
    fig.savefig('bar_per_sample_comparsion_plot_{}.pdf'.format(dataset), dpi=300)

if __name__ == '__main__':
    plot_bar_per_sample_com('human_gut', diff_label=[-5,0,5,10,15,20])
    plot_bar_per_sample_com('dog_gut', diff_label=[-5,0,5,10,15,20])
    plot_bar_per_sample_com('ocean', diff_label=[-5,0,5,10,15,20])
    plot_bar_per_sample_com('soil', diff_label=[-5,0,5,10,15,20])

