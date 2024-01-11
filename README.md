dHICA
===============

Discriminative histone imputation using chromatin accessibility

Cloud Computing Service:
-------------------------
We provide a computational gateway to run dHICA on GPU server, the users don't need to install any software, only upload the bigWig files and wait for the results, it is simple and easy. Please click the link to try this site:

https://dreg.dnasequence.org/


![Hi](https://github.com/Danko-Lab/dREG/raw/master/dreg-gateway.png?v=4&s=200 "dREG gateway")


Abstract:
--------
**Motivation**: Histone modifications (HMs) play a crucial role in various biological processes, including transcription, replication, and DNA repair, and they have a significant impact on chromatin structure. However, annotating HMs across different cell types using experimental approaches alone is impractical due to cost and time constraints. Therefore, computational techniques are valuable for predicting HMs, complementing experimental methods, and enhancing our understanding of DNA characteristics and gene expression patterns.

**Results**: In this study, we introduce a deep learning model called discriminative histone imputation using chromatin accessibility (dHICA). dHICA combines extensive DNA sequence information with chromatin accessibility data, utilizing the Transformer architecture. This allows dHICA to have a large receptive field, and more cell-type specific information, which can not only incorporate distal information across the entire genome, but also improve predictive accuracy and interpretability. Comparing performance with other models, we found that dHICA exhibits superior performance in both peak calling and signal predictions, especially in biologically salient regions such as gene elements. dHICA serves as a crucial tool for advancing our understanding of chromatin dynamics, offering enhanced predictive capabilities and interpretability, particularly in critical biological regions such as gene elements. 

Data preparation: 
==========================

dHICA takes bigWig files with single strands（ATAC-seq or DNase-seq） as the input. Performing data normalization beforehand is **NOT** recommended.


Download instruction: 
==========================
Install dHICA
------------
dHICA's source code is availiable on GitHub (https://github.com/wzhy2000/dHICA).  

Required software
-----------------
* Python 3.9
* Tensorflow 2.13.0 (https://www.tensorflow.org/)
* pyBigwig (https://github.com/deeptools/pyBigWig)


Get the dHICA models
-------------------
Pre-trained model that can be used to predict dREG scores across the genome is availiable here.
If you are failed to download the model files, please contact us.

Usage instruction:
===================

## 1 Predict HMs with Pre-trained Models
First you need to make sure you have download the our Pre-trained Models. The type of ATAC-seq should be fold over change, and that of DNase-seq should be read-depth normalized signal.

    python predict.py -m model_path  -o output_path 
                        --atac bw_path --op output_prefix --ref ref_genome.fa

    model_path      -- The path to the pre-trained model(DNase or ATAC).
    output_path     -- The path to save the output files.
    ref_genome.fa   -- Reference genome.(hg19 or mm10)
    bw_path         -- Read counts (not normalized) formatted as a bigWig or bw file.
    output_prefix   -- The prefix of the output file.

After the command execution, you will obtain 10 BigWig (bw) files for different histone modifications. You can visualize and inspect these files using tools such as IGV (Integrative Genomics Viewer) or Genome Browser. Alternatively, you can utilize the Python package pyBigWig to read the signal data.

## 2 Training Your Own Model
### 1） Generate Dataset
The type of ATAC-seq should be fold over change, and that of DNase-seq should be read-depth normalized signal.
    python dHICA_data.py -l seq_length --local -o ouput_path --seq seq_type ref_genome.fa target_HM.txt
    
    seq_length      -- [default=131072] Sequence length.
    ouput_path      -- The path to save the output files.
    seq_type        -- The sequencing type(ATAC or DNase).
    ref_genome.fa   -- Reference genome.(hg19 or mm10)
    target_HM.txt   -- The HMs target file.


### 2） Train Model 
    python train.py

**Notice:** 
That command takes more than 8 hours (depends on size of the train datasets) to execute on NVIDA 3090 GPU. Due to very long computational time, we don't suggest to run on CPU nodes.

### 3） Calculate Performance
    python correlation.py -a dHICA.bw -b ref.bw -p resolution

    dHICA.bw    -- dHICA's output.
    ref.bw      -- the HMs target.
    resolution  -- a figure(128, 1000 or 10000) or a bed file.


