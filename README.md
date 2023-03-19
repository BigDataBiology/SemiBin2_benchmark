# SemiBin2_benchmark



Code for SemiBin2 benchmark

### Benchmark  ###

#### Short Read ####

##### VAMB #####

```bash
vamb --outdir VAMB --fasta input.fasta --bamfiles *.bam --minfasta 200000 -m 2000 --cuda -o :
```

##### SemiBin #####

```ba
SemiBin multi_easy_bin -i input.fasta -b *.bam --orf-finder fraggenescan -s : -o SemiBin
```

##### SemiBin2 #####

```bash
SemiBin multi_easy_bin -i input.fasta -b *.bam --orf-finder fraggenescan -s : --self-supervised -o SemiBin2
```

#### Long Read ####

##### LRBinner #####

```ba
~/LRBinner/LRBinner contigs --reads-path input.fq --threads 64 --resume --output LRBinner --contigs input.fasta
```

##### MetaBAT2 #####

```bash
jgi_summarize_bam_contig_depths --outputDepth input.fasta.depth.txt --minContigLength 1000 --minContigDepth 1 input.bam --percentIdentity 50

metabat2 --seed 1234 -t 48 --inFile input.fasta --outFile Metabat2/Metabat2 --abdFile input.fasta.depth.txt
```

##### VAMB #####

```bash
vamb --outdir VAMB --fasta input.fasta --bamfiles *.bam --minfasta 200000 -m 2000 --cuda 
```

##### SemiBin #####

```bash
SemiBin single_easy_bin -i input.fasta -b *.bam --orf-finder fraggenescan -o SemiBin
```

##### SemiBin2 #####

```bash
SemiBin single_easy_bin -i input.fasta -b *.bam --orf-finder fraggenescan -o SemiBin2 --self-supervised
```

##### GraphMB #####

```bash
graphmb --assembly contig_dir --outdir output --markers checkm_edges/storage/marker_gene_stats.tsv --cuda
```

##### MetaDecoder #####

```bash
metadecoder coverage -s input.sam -o METADECODER.COVERAGE
metadecoder seed --threads 50 -f input.fasta -o METADECODER.SEED
metadecoder cluster -f input.fasta -c METADECODER.COVERAGE -s METADECODER.SEED -o METADECODER
```

##### MetaBinner #####

```bash
use  gen_coverage_file.sh to generate mb2_master_depth.txt
cat mb2_master_depth.txt | cut -f -1,4- > coverage_profile.tsv
cat mb2_master_depth.txt | awk '{if ($2>1000) print $0 }' | cut -f -1,4- > coverage_profile_f1k.tsv

python Filter_tooshort.py input.fasta 1000
python gen_kmer.py input_1000.fa 1000 4
bash run_metabinner.sh -a input_1000.fa -o output -d coverage_profile_f1k.tsv -k input_1000_kmer_4_f1000.csv -p MetaBinner-master -t 32
```

##### CONCOCT #####

```bash
cut_up_fasta.py input.fa -c 10000 -o 0 --merge_last -b contigs_10K.bed > contigs_10K.fa
concoct_coverage_table.py contigs_10K.bed input.bam > coverage_table.tsv
concoct --composition_file contigs_10K.fa --coverage_file coverage_table.tsv -b concoct_output
merge_cutup_clustering.py concoct_output/clustering_gt1000.csv > concoct_output/clustering_merged.csv
mkdir concoct_output/fasta_bins
extract_fasta_bins.py input.fa concoct_output/clustering_merged.csv --output_path concoct_output/fasta_bins
```

#### Evaluation ####

##### Amber #####

```bash
python amber.py -g gsa_mapping.binning \
-l "Method" \
Method/bins.tsv \
-o output_dir/
```

##### CheckM #####

```bash
checkm lineage_wf -x fa -t 48 -f checkm_result.txt --tab_table bins_output checkm_output
```

##### CheckM2 #####

```bash
checkm2 predict --threads 30 --input bins_output  --output-directory checkm2_output -x .fa
```

##### GUNC #####

```bash
mkdir GUNC
gunc run -d bins_output -o GUNC -r ~/gunc_path/gunc_db_2.0.4.dmnd -t 32
```













