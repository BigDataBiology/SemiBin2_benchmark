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













