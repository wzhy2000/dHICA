#!/usr/bin/env python
from __future__ import print_function

from optparse import Option, OptionParser
import collections
import gzip
import heapq
import json
import math
import pdb
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
import pandas as pd

import genome
import util

try:
    import slurm
except ModuleNotFoundError:
    pass

'''
basenji_data.py

Compute model sequences from the genome, extracting DNA coverage values.
'''


def first_five(path):
    with open(path, 'r') as r_obj:
        sections = r_obj.readlines()
    sections = [section.split()[:5] for section in sections]
    train_set = []
    valid_set = []
    test_set = []
    train_chr22 = []
    valid_chr22 = []
    test_chr22 = []
    for section in sections:
        if section[0] == 'chrY':#去掉Y染色体
            continue
        if not (section[0] == 'chr22'):
            if section[-2] == 'train':
                train_set.append({'chr': section[0], 'start': section[1], 'end': section[2], 'protype': section[4]})
            elif section[-2] == 'valid':
                valid_set.append({'chr': section[0], 'start': section[1], 'end': section[2], 'protype': section[4]})
            else:
                test_set.append({'chr': section[0], 'start': section[1], 'end': section[2], 'protype': section[4]})
        else:
            if section[-2] == 'train':
                train_chr22.append({'chr': section[0], 'start': section[1], 'end': section[2], 'protype': section[4]})
            elif section[-2] == 'valid':
                valid_chr22.append({'chr': section[0], 'start': section[1], 'end': section[2], 'protype': section[4]})
            else:
                test_chr22.append({'chr': section[0], 'start': section[1], 'end': section[2], 'protype': section[4]})
    return train_set, valid_set, test_set, train_chr22, valid_chr22, test_chr22


def add_protype(path, protype, chromsome):
    with open(path, 'r') as r_obj:
        sections = r_obj.readlines()
    sections = [section.split()[:4] for section in sections]
    with open(path, 'w') as w_obj:
        if chromsome is None:
            for section in sections:
                w_obj.write(
                    section[0] + '\t' + section[1] + '\t' + section[2] + '\t' + section[3] + '\t' + protype + '\n')
        else:
            for section in sections:
                if section[0] == chromsome:
                    w_obj.write(
                        section[0] + '\t' + section[1] + '\t' + section[2] + '\t' + section[3] + '\t' + protype + '\n')


def split_contigs(chrom_segments, gaps_file):
    """ Split the assembly up into contigs defined by the gaps.

      Args:
        chrom_segments: dict mapping chromosome names to lists of (start,end)
        gaps_file: file specifying assembly gaps

      Returns:
        chrom_segments: same, with segments broken by the assembly gaps.
      """

    chrom_events = {}

    # add known segments
    for chrom in chrom_segments:
        if len(chrom_segments[chrom]) > 1:
            print(
                "I've made a terrible mistake...regarding the length of chrom_segments[%s]"
                % chrom,
                file=sys.stderr)
            exit(1)
        cstart, cend = chrom_segments[chrom][0]
        chrom_events.setdefault(chrom, []).append((cstart, 'Cstart'))
        chrom_events[chrom].append((cend, 'cend'))

    # add gaps
    for line in open(gaps_file):
        a = line.split()
        chrom = a[0]
        gstart = int(a[1])
        gend = int(a[2])

        # consider only if its in our genome
        if chrom in chrom_events:
            # if语句是后加的
            # if gend < cend:
            chrom_events[chrom].append((gstart, 'gstart'))
            chrom_events[chrom].append((gend, 'Gend'))

    for chrom in chrom_events:
        # sort
        chrom_events[chrom].sort()

        # read out segments
        chrom_segments[chrom] = []
        # print(chrom_events['chr1'])
        for i in range(len(chrom_events[chrom]) - 1):
            pos1, event1 = chrom_events[chrom][i]
            pos2, event2 = chrom_events[chrom][i + 1]

            event1 = event1.lower()
            event2 = event2.lower()

            shipit = False
            if event1 == 'cstart' and event2 == 'cend':
                shipit = True
            elif event1 == 'cstart' and event2 == 'gstart':
                shipit = True
            elif event1 == 'gend' and event2 == 'gstart':
                shipit = True
            elif event1 == 'gend' and event2 == 'cend':
                shipit = True
            elif event1 == 'gstart' and event2 == 'gend':
                pass
            else:

                print(
                    "I'm confused by this event ordering: %s - %s" % (event1, event2),
                    file=sys.stderr)
                print(pos1, pos2)
                exit(1)

            if shipit and pos1 < pos2:
                chrom_segments[chrom].append((pos1, pos2))

    return chrom_segments


################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='blacklist_bed',
                      help='Set blacklist nucleotides to a baseline value.')
    parser.add_option('--break', dest='break_t',
                      default=None, type='int',
                      help='Break in half contigs above length [Default: %default]')
    parser.add_option('-c', '--crop', dest='crop_bp',
                      default=0, type='int',
                      help='Crop bp off each end [Default: %default]')
    parser.add_option('-d', dest='sample_pct',
                      default=1.0, type='float',
                      help='Down-sample the segments')
    parser.add_option('-f', dest='folds',
                      default=None, type='int',
                      help='Generate cross fold split [Default: %default]')
    parser.add_option('-g', dest='gaps_file',
                      help='Genome assembly gaps BED [Default: %default]')
    parser.add_option('-i', dest='interp_nan',
                      default=False, action='store_true',
                      help='Interpolate NaNs [Default: %default]')
    parser.add_option('-l', dest='seq_length',
                      default=131072, type='int',
                      help='Sequence length [Default: %default]')
    parser.add_option('--limit', dest='limit_bed',
                      help='Limit to segments that overlap regions in a BED file')
    parser.add_option('--local', dest='run_local',
                      default=False, action='store_true',
                      help='Run jobs locally as opposed to on SLURM [Default: %default]')
    parser.add_option('-o', dest='out_dir',
                      default='data_out',
                      help='Output directory [Default: %default]')
    parser.add_option('-p', dest='processes',
                      default=None, type='int',
                      help='Number parallel processes [Default: %default]')
    parser.add_option('--peaks', dest='peaks_only',
                      default=False, action='store_true',
                      help='Create contigs only from peaks [Default: %default]')
    parser.add_option('-r', dest='seqs_per_tfr',
                      default=256, type='int',
                      help='Sequences per TFRecord file [Default: %default]')
    parser.add_option('--restart', dest='restart',
                      default=False, action='store_true',
                      help='Continue progress from midpoint. [Default: %default]')
    parser.add_option('--seed', dest='seed',
                      default=44, type='int',
                      help='Random seed [Default: %default]')
    parser.add_option('--snap', dest='snap',
                      default=1, type='int',
                      help='Snap sequences to multiple of the given value [Default: %default]')
    parser.add_option('--st', '--split_test', dest='split_test',
                      default=False, action='store_true',
                      help='Exit after split. [Default: %default]')
    parser.add_option('--stride', '--stride_train', dest='stride_train',
                      default=1., type='float',
                      help='Stride to advance train sequences [Default: seq_length]')
    parser.add_option('--stride_test', dest='stride_test',
                      default=1., type='float',
                      help='Stride to advance valid and test sequences [Default: seq_length]')
    parser.add_option('-t', dest='test_pct_or_chr',
                      default=0.1, type='str',
                      help='Proportion of the data for testing [Default: %default]')
    parser.add_option('-u', dest='umap_bed',
                      help='Unmappable regions in BED format')
    parser.add_option('--umap_t', dest='umap_t',
                      default=0.5, type='float',
                      help='Remove sequences with more than this unmappable bin % [Default: %default]')
    parser.add_option('--umap_clip', dest='umap_clip',
                      default=1, type='float',
                      help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
    parser.add_option('--umap_tfr', dest='umap_tfr',
                      default=False, action='store_true',
                      help='Save umap array into TFRecords [Default: %default]')
    parser.add_option('-w', dest='pool_width',
                      default=128, type='int',
                      help='Sum pool width [Default: %default]')
    parser.add_option('-v', dest='valid_pct_or_chr',
                      default=0.1, type='str',
                      help='Proportion of the data for validation [Default: %default]')

    parser.add_option('--pro', dest='protype',
                      default='G1', type='str',
                      help='Protype of ro-seq data [Default: %default]')

    parser.add_option('--chr', dest='chromsome',
                      default=None, type='str',
                      help='the Chromsome you choosed [Default: %default]')

    parser.add_option('--mouse', dest='mouse',
                      default=False, action='store_true',
                      help='Is mouse?')
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide FASTA, sample coverage and proseq labels and paths.')
    else:
        fasta_file = args[0]
        targets_file = args[1]
        proseq_id_file = args[2]

    random.seed(options.seed)
    np.random.seed(options.seed)
    # print(options.snap)
    # exit()
    # option.break_t默认是none
    if options.break_t is not None and options.break_t < options.seq_length:
        print('Maximum contig length --break cannot be less than sequence length.', file=sys.stderr)
        exit(1)

    # transform proportion strides to base pairs
    if options.stride_train <= 1:
        print('stride_train %.f' % options.stride_train, end='')
        options.stride_train = options.stride_train * options.seq_length
        print(' converted to %f' % options.stride_train)
    options.stride_train = int(np.round(options.stride_train))

    if options.stride_test <= 1:
        if options.folds is None:
            print('stride_test %.f' % options.stride_test, end='')
            options.stride_test = options.stride_test * options.seq_length
            print(' converted to %f' % options.stride_test)
    options.stride_test = int(np.round(options.stride_test))

    # check snap
    if options.snap is not None:
        if np.mod(options.seq_length, options.snap) != 0:
            raise ValueError('seq_length must be a multiple of snap')
        if np.mod(options.stride_train, options.snap) != 0:
            raise ValueError('stride_train must be a multiple of snap')
        if np.mod(options.stride_test, options.snap) != 0:
            raise ValueError('stride_test must be a multiple of snap')

    # setup output directory
    if os.path.isdir(options.out_dir) and not options.restart:
        print('Remove output directory %s or use --restart option.' % options.out_dir)
        exit(1)
    elif not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # read target datasets
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')

    ################################################################
    # define genomic contigs(连续的reads组合在一块就是contig（重叠群）)
    ################################################################
    if not options.restart:
        # Load genome segments from either a FASTA file or chromosome length table
        chrom_contigs = genome.load_chromosomes(fasta_file)
        # print(chrom_contigs)
        '''
        {'chr1': [(0, 248956422)], 'chr10': [(0, 133797422)], 'chr11': [(0, 135086622)], 'chr11_KI270721v1_random'...}
        '''
        # remove gaps
        if options.gaps_file:
            # 染色体上的一些没有和重叠群重叠的片段{'chr1': [(1,2)， （3,4), ...,]}
            chrom_contigs = split_contigs(chrom_contigs, options.gaps_file)

            # print(chrom_contigs)
            # exit()
        # print(len(chrom_contigs))
        # ditch the chromosomes for contigs

        contigs = []
        for chrom in chrom_contigs:
            # if是后加的
            if (not chrom.startswith('chrUn')) and ('_' not in chrom) and (chrom != 'chrM') and (
                    chrom != 'chrEBV') and (chrom != 'chrMT') and (chrom != 'chrY'):
                contigs += [Contig(chrom, ctg_start, ctg_end)
                            for ctg_start, ctg_end in chrom_contigs[chrom]]

        print(len(contigs))
        for ctg in contigs:
            print(ctg.chr)
        # exit(1)

        # limit to a BED file
        # Limit to contigs overlapping the given BED
        if options.limit_bed is not None:
            contigs = limit_contigs(contigs, options.limit_bed)

        # limit to peaks
        if options.peaks_only:
            peaks_bed = curate_peaks(targets_df, options.out_dir, options.pool_width, options.crop_bp)
            contigs = limit_contigs(contigs, peaks_bed)

        # filter for large enough
        # 把比seq_length长的重叠群留下
        seq_tlength = options.seq_length - 2 * options.crop_bp
        contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_tlength]

        # print(len(contigs))
        # break up large contigs
        # break_t = 786432
        if options.break_t is not None:
            contigs = break_large_contigs(contigs, options.break_t)

        # print(len(contigs))
        # print contigs to BED file
        ctg_bed_file = '%s/contigs_break.bed' % options.out_dir
        write_seqs_bed(ctg_bed_file, contigs)

        # print(contigs)

    ################################################################
    # divide between train/valid/test
    ################################################################
    # label folds
    if options.folds is not None:
        fold_labels = ['fold%d' % fi for fi in range(options.folds)]
        num_folds = options.folds
    else:
        fold_labels = ['train', 'valid', 'test']
        num_folds = 3

    ctg_bed_file = '%s/contigs_break.bed' % options.out_dir
    ctg_bed_out = open(ctg_bed_file, 'w')

    if not options.restart:
        if options.folds is not None:
            # divide by fold pct
            fold_contigs = divide_contigs_folds(contigs, options.folds)

        else:
            try:
                # convert to float pct
                valid_pct = float(options.valid_pct_or_chr)
                test_pct = float(options.test_pct_or_chr)
                print(valid_pct, test_pct) #default均为0.1
                assert (0 <= valid_pct <= 1)
                assert (0 <= test_pct <= 1)

                # divide by pct
                # print(len(contigs))
                fold_contigs = divide_contigs_pct(contigs, test_pct, valid_pct)
                # print(len(fold_contigs[0]))
            except (ValueError, AssertionError):
                # divide by chromatins
                valid_chrs = options.valid_pct_or_chr.split(',')
                test_chrs = options.test_pct_or_chr.split(',')
                fold_contigs = divide_contigs_chr(contigs, test_chrs, valid_chrs)

        # rejoin broken contigs within set
        for fi in range(len(fold_contigs)):
            # print(len(fold_contigs[fi]))
            # exit(1)
            fold_contigs[fi] = rejoin_large_contigs(fold_contigs[fi])

        # write labeled contigs to BED file
        ctg_bed_file = '%s/contigs.bed' % options.out_dir
        ctg_bed_out = open(ctg_bed_file, 'w')
        for fi in range(len(fold_contigs)):
            for ctg in fold_contigs[fi]:
                line = '%s\t%d\t%d\t%s' % (ctg.chr, ctg.start, ctg.end, fold_labels[fi])
                # fold_labels = ['train', 'valid', 'test']
                print(line, file = ctg_bed_out)
        ctg_bed_out.close()
        # exit()
    if options.split_test:
        exit()

    ################################################################
    # define model sequences
    ################################################################
    if not options.restart:
        fold_mseqs = []
        for fi in range(num_folds):#num_fold = 3 (train,test,valid)
            if fold_labels[fi] in ['valid', 'test']: #train和test、valid的stride不一样
                stride_fold = options.stride_test
            else:
                stride_fold = options.stride_train

            # stride sequences across contig
            # print(seq_tlength, stride_fold, options.snap)
            # contig_sequences: Break up a list of Contig's into a list of ModelSeq's.
            fold_mseqs_fi = contig_sequences(fold_contigs[fi], seq_tlength,
                                             stride_fold, options.snap, fold_labels[fi])
            fold_mseqs.append(fold_mseqs_fi)

            # shuffle
            random.shuffle(fold_mseqs[fi])

            # down-sample
            if options.sample_pct < 1.0:
                fold_mseqs[fi] = random.sample(fold_mseqs[fi], int(options.sample_pct * len(fold_mseqs[fi])))
        # for i in range(len(fold_mseqs)):
        #   print(len(fold_mseqs[i]))
        # exit()
        # merge into one list
        mseqs = [ms for fm in fold_mseqs for ms in fm]

    ################################################################
    # mappability
    ################################################################

    if not options.restart:
        if options.umap_bed is not None:
            if shutil.which('bedtools') is None:
                print('Install Bedtools to annotate unmappable sites', file=sys.stderr)
                exit(1)

            # annotate unmappable positions
            mseqs_unmap = annotate_unmap(mseqs, options.umap_bed,
                                         seq_tlength, options.pool_width)

            # filter unmappable
            mseqs_map_mask = (mseqs_unmap.mean(axis=1, dtype='float64') < options.umap_t)
            mseqs = [mseqs[i] for i in range(len(mseqs)) if mseqs_map_mask[i]]
            mseqs_unmap = mseqs_unmap[mseqs_map_mask, :]

            # write to file
            unmap_npy = '%s/mseqs_unmap.npy' % options.out_dir
            np.save(unmap_npy, mseqs_unmap)

        # write sequences to BED
        seqs_bed_file = '%s/sequences.bed' % options.out_dir
        write_seqs_bed(seqs_bed_file, mseqs, True)
        # exit()

    else:
        # read from directory
        seqs_bed_file = '%s/sequences.bed' % options.out_dir
        unmap_npy = '%s/mseqs_unmap.npy' % options.out_dir
        mseqs = []
        fold_mseqs = []
        # for fi in range(num_folds):
        fold_mseqs.append([])
        for line in open(seqs_bed_file):
            a = line.split()
            msg = ModelSeq(a[0], int(a[1]), int(a[2]), a[3])
            mseqs.append(msg)
            if a[3] == 'train':
                fi = 0
            elif a[3] == 'valid':
                fi = 1
            elif a[3] == 'test':
                fi = 2
            else:
                fi = int(a[3].replace('fold', ''))
            fold_mseqs[fi].append(msg)

    # ------------------------------------------------------------------------------------------------------------------

    add_protype(seqs_bed_file, options.protype, options.chromsome)


    #add_protype(seqs_bed_file)
    '''
    proseq_path = '/home/dut_ww/enformer/data/k562/all_atac.bed'
    #应该是把g1，g2···g7的sequence都合并到这一个all_proseq.bed中！！！
    #将四个atac文件分成atac1...atac4
    train_seq, valid_seq, test_seq, train_chr22_seq, valid_chr22_seq, test_chr22_seq = first_five(proseq_path)
    random.shuffle(train_seq)
    random.shuffle(valid_seq)
    random.shuffle(test_seq)
    random.shuffle(train_chr22_seq)
    random.shuffle(valid_chr22_seq)
    random.shuffle(test_chr22_seq)

    with open(seqs_bed_file, 'w') as w_obj:
        for item in train_seq:
            w_obj.write(item['chr'] + '\t' + item['start'] + '\t' + item['end'] + '\t' + 'train' + '\t' + item['protype'] + '\n')
        for item in valid_seq:
            w_obj.write(item['chr'] + '\t' + item['start'] + '\t' + item['end'] + '\t' + 'valid' + '\t' + item['protype'] + '\n')
        for item in test_seq:
            w_obj.write(item['chr'] + '\t' + item['start'] + '\t' + item['end'] + '\t' + 'test' + '\t' + item['protype'] + '\n')
  
    out_chr22 = '%s/chr22.bed' % options.out_dir
    with open(out_chr22, 'w') as w_obj:
        for item in train_chr22_seq:
            w_obj.write(item['chr'] + '\t' + item['start'] + '\t' + item['end'] + '\t' + 'train' + '\t' + item['protype'] + '\n')
        for item in valid_chr22_seq:
            w_obj.write(item['chr'] + '\t' + item['start'] + '\t' + item['end'] + '\t' + 'valid' + '\t' + item['protype'] + '\n')
        for item in test_chr22_seq:
            w_obj.write(item['chr'] + '\t' + item['start'] + '\t' + item['end'] + '\t' + 'test' + '\t' + item['protype'] + '\n')
  
    mseqs = []
    fold_mseqs = []
    for fi in range(num_folds):
      fold_mseqs.append([])
    for line in open(seqs_bed_file):
      a = line.split()
      msg = ModelSeq(a[0], int(a[1]), int(a[2]), a[3])
      mseqs.append(msg)
      if a[3] == 'train':
        fi = 0
      elif a[3] == 'valid':
        fi = 1
      elif a[3] == 'test':
        fi = 2
      else:
        fi = int(a[3].replace('fold',''))
      fold_mseqs[fi].append(msg)
    '''
# -------------------------------------------------------------------------------------------------

    # exit()
    ################################################################
    # read sequence coverage values
    ################################################################
    seqs_cov_dir = '%s/seqs_cov' % options.out_dir
    if not os.path.isdir(seqs_cov_dir):
        os.mkdir(seqs_cov_dir)

    read_jobs = []
    for ti in range(targets_df.shape[0]):
        genome_cov_file = targets_df['file'].iloc[ti]
        seqs_cov_stem = '%s/%d' % (seqs_cov_dir, ti)
        seqs_cov_file = '%s.h5' % seqs_cov_stem

        clip_ti = None
        if 'clip' in targets_df.columns:
            clip_ti = targets_df['clip'].iloc[ti]

        clipsoft_ti = None
        if 'clip_soft' in targets_df.columns:
            clipsoft_ti = targets_df['clip_soft'].iloc[ti]

        scale_ti = 1
        if 'scale' in targets_df.columns:
            scale_ti = targets_df['scale'].iloc[ti]

        if options.restart and os.path.isfile(seqs_cov_file):
            print('Skipping existing %s' % seqs_cov_file, file=sys.stderr)
        else:
            cmd = 'python /local/ww/enformer/data_code/dnase/basenji_data_read.py'
            # cmd += ' --crop %d' % options.crop_bp
            cmd += ' -w %d' % options.pool_width
            cmd += ' -u %s' % targets_df['sum_stat'].iloc[ti]
            if clip_ti is not None:
                cmd += ' -c %f' % clip_ti
            if clipsoft_ti is not None:
                cmd += ' --clip_soft %f' % clipsoft_ti
            cmd += ' -s %f' % scale_ti
            if options.blacklist_bed:
                cmd += ' -b %s' % options.blacklist_bed
            if options.interp_nan:
                cmd += ' -i'
            cmd += ' %s' % genome_cov_file # fasta文件
            cmd += ' %s' % seqs_bed_file  # sequence.bed
            cmd += ' %s' % seqs_cov_file
            # print(seqs_cov_file)
            # print(cmd)
            # print(zzx)
            if options.run_local:
                # breaks on some OS
                # cmd += ' &> %s.err' % seqs_cov_stem
                read_jobs.append(cmd)
            else:
                j = slurm.Job(cmd,
                              name='read_t%d' % ti,
                              out_file='%s.out' % seqs_cov_stem,
                              err_file='%s.err' % seqs_cov_stem,
                              queue='standard', mem=15000, time='12:0:0')
                read_jobs.append(j)

    if options.run_local:
        util.exec_par(read_jobs, options.processes, verbose=True)
    else:
        slurm.multi_run(read_jobs, options.processes, verbose=True,
                        launch_sleep=1, update_sleep=5)
    # exit()
    ################################################################
    # write TF Records
    ################################################################
    # copy targets file
    shutil.copy(targets_file, '%s/targets.txt' % options.out_dir)

    # initialize TF Records dir
    tfr_dir = '%s/tfrecords' % options.out_dir
    if not os.path.isdir(tfr_dir):
        os.mkdir(tfr_dir)

    write_jobs = []

    #fold_labels = ['train']
    for fold_set in fold_labels:
        fold_set_indexes = [i for i in range(len(mseqs)) if mseqs[i].label == fold_set] # 挑出mseq里面分别挑出train数据，以及其对应的index
        print(fold_set_indexes)
        print(fold_set)
        print(len(mseqs))
        print(fold_set_indexes[0])
        fold_set_start = fold_set_indexes[0]
        fold_set_end = fold_set_indexes[-1] + 1

        tfr_i = 0
        tfr_start = fold_set_start
        tfr_end = min(tfr_start + options.seqs_per_tfr, fold_set_end)

        while tfr_start <= fold_set_end:
            tfr_stem = '%s/%s-%d' % (tfr_dir, fold_set, tfr_i)
            cmd = 'python /local/ww/enformer/data_code/dnase/basenji_data_write.py'
            cmd += ' -s %d' % tfr_start
            cmd += ' -e %d' % tfr_end
            cmd += ' --umap_clip %f' % options.umap_clip # default = 1
            cmd += ' -x %d' % options.crop_bp
            # if options.zzx:
            #   cmd += ' --zzx'
            if options.mouse:
                cmd += ' --mouse'
            if options.umap_tfr:
                cmd += ' --umap_tfr'
            if options.umap_bed is not None:
                cmd += ' -u %s' % unmap_npy

            cmd += ' %s' % fasta_file
            cmd += ' %s' % seqs_bed_file
            cmd += ' %s' % seqs_cov_dir
            cmd += ' %s' % proseq_id_file # 其实basenji_data_write.py中没有用到这个txt文件
            cmd += ' %s.tfr' % tfr_stem

            # print(cmd)
            # print(zzx)
            if options.run_local:
                # breaks on some OS
                # cmd += ' &> %s.err' % tfr_stem
                write_jobs.append(cmd)
            else:
                j = slurm.Job(cmd,
                              name='write_%s-%d' % (fold_set, tfr_i),
                              out_file='%s.out' % tfr_stem,
                              err_file='%s.err' % tfr_stem,
                              queue='standard', mem=15000, time='12:0:0')
                write_jobs.append(j)

            # update
            tfr_i += 1
            tfr_start += options.seqs_per_tfr
            tfr_end = min(tfr_start + options.seqs_per_tfr, fold_set_end)
    # print(write_jobs)
    # exit()
    if options.run_local:
        util.exec_par(write_jobs, options.processes, verbose=True)
    else:
        slurm.multi_run(write_jobs, options.processes, verbose=True,
                        launch_sleep=1, update_sleep=5)

    ################################################################
    # stats
    ################################################################
    # with open(proseq_id_file, 'r') as r_obj:
    #   proseq_id = r_obj.readlines()
    # num_proseq = len(proseq_id)
    stats_dict = {}
    stats_dict['num_targets'] = targets_df.shape[0]
    stats_dict['num_atac'] = 1
    # print('-'*50)
    # print(options.seq_length)
    stats_dict['atac_length'] = (options.seq_length + 2 * 32768)  # // options.pool_width
    stats_dict['seq_length'] = options.seq_length + 2 * 32768
    # print('-'*50)

    stats_dict['pool_width'] = options.pool_width  # 128
    stats_dict['crop_bp'] = options.crop_bp

    target_length = options.seq_length + 2 * 32768 - 2 * options.crop_bp
    target_length = target_length // options.pool_width
    stats_dict['target_length'] = target_length - (196608 - 114688) // options.pool_width

    stats_dict['%s_seqs' % fold_labels[0]] = len(fold_mseqs[0])
    stats_dict['%s_seqs' % fold_labels[1]] = len(fold_mseqs[1])
    stats_dict['%s_seqs' % fold_labels[2]] = len(fold_mseqs[2])

    with open('%s/statistics.json' % options.out_dir, 'w') as stats_json_out:
        json.dump(stats_dict, stats_json_out, indent=4) # indent-间隔符的长度


################################################################################
def annotate_unmap(mseqs, unmap_bed, seq_length, pool_width):
    """ Intersect the sequence segments with unmappable regions
           and annoate the segments as NaN to possible be ignored.

      Args:
        mseqs: list of ModelSeq's
        unmap_bed: unmappable regions BED file
        seq_length: sequence length (after cropping)
        pool_width: pooled bin width

      Returns:
        seqs_unmap: NxL binary NA indicators
      """

    # print sequence segments to file
    seqs_temp = tempfile.NamedTemporaryFile()
    seqs_bed_file = seqs_temp.name
    write_seqs_bed(seqs_bed_file, mseqs)

    # hash segments to indexes
    chr_start_indexes = {}
    for i in range(len(mseqs)):
        chr_start_indexes[(mseqs[i].chr, mseqs[i].start)] = i

    # initialize unmappable array
    pool_seq_length = seq_length // pool_width
    seqs_unmap = np.zeros((len(mseqs), pool_seq_length), dtype='bool')

    # intersect with unmappable regions
    p = subprocess.Popen(
        'bedtools intersect -wo -a %s -b %s' % (seqs_bed_file, unmap_bed),
        shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        line = line.decode('utf-8')
        a = line.split()

        seq_chrom = a[0]
        seq_start = int(a[1])
        seq_end = int(a[2])
        seq_key = (seq_chrom, seq_start)

        unmap_start = int(a[4])
        unmap_end = int(a[5])

        overlap_start = max(seq_start, unmap_start)
        overlap_end = min(seq_end, unmap_end)

        pool_seq_unmap_start = math.floor((overlap_start - seq_start) / pool_width)
        pool_seq_unmap_end = math.ceil((overlap_end - seq_start) / pool_width)

        # skip minor overlaps to the first
        first_start = seq_start + pool_seq_unmap_start * pool_width
        first_end = first_start + pool_width
        first_overlap = first_end - overlap_start
        if first_overlap < 0.1 * pool_width:
            pool_seq_unmap_start += 1

        # skip minor overlaps to the last
        last_start = seq_start + (pool_seq_unmap_end - 1) * pool_width
        last_overlap = overlap_end - last_start
        if last_overlap < 0.1 * pool_width:
            pool_seq_unmap_end -= 1

        seqs_unmap[chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end] = True
        assert (seqs_unmap[chr_start_indexes[seq_key],
                pool_seq_unmap_start:pool_seq_unmap_end].sum() == pool_seq_unmap_end - pool_seq_unmap_start)

    return seqs_unmap


################################################################################
def break_large_contigs(contigs, break_t, verbose=False):
    """Break large contigs in half until all contigs are under
       the size threshold."""

    # initialize a heapq of contigs and lengths
    contig_heapq = []
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start
        heapq.heappush(contig_heapq, (-ctg_len, ctg))

    ctg_len = break_t + 1
    while ctg_len > break_t:

        # pop largest contig
        ctg_nlen, ctg = heapq.heappop(contig_heapq)
        ctg_len = -ctg_nlen

        # if too large
        if ctg_len > break_t:
            if verbose:
                print('Breaking %s:%d-%d (%d nt)' % (ctg.chr, ctg.start, ctg.end, ctg_len))

            # break in two
            ctg_mid = ctg.start + ctg_len // 2

            try:
                ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)
            except AttributeError:
                ctg_left = Contig(ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.chr, ctg_mid, ctg.end)

            # add left
            ctg_left_len = ctg_left.end - ctg_left.start
            heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

            # add right
            ctg_right_len = ctg_right.end - ctg_right.start
            heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

    # return to list
    contigs = [len_ctg[1] for len_ctg in contig_heapq]

    return contigs


################################################################################
def contig_sequences(contigs, seq_length, stride, snap=1, label=None):
    ''' Break up a list of Contig's into a list of ModelSeq's. '''
    mseqs = []
    for ctg in contigs:
        seq_start = int(np.ceil(ctg.start / snap) * snap)
        seq_end = seq_start + seq_length

        while seq_end <= ctg.end:
            # record sequence
            mseqs.append(ModelSeq(ctg.chr, seq_start, seq_end, label))

            # update
            seq_start += stride
            seq_end += stride
        mseqs.append(ModelSeq(ctg.chr, ctg.end - seq_length, ctg.end, label))

    return mseqs


################################################################################
def curate_peaks(targets_df, out_dir, pool_width, crop_bp):
    """Merge all peaks, round to nearest pool_width, and add cropped bp."""

    # concatenate and extend peaks
    cat_bed_file = '%s/peaks_cat.bed' % out_dir
    cat_bed_out = open(cat_bed_file, 'w')
    for bed_file in targets_df.file:
        if bed_file[-3:] == '.gz':
            bed_in = gzip.open(bed_file, 'rt')
        else:
            bed_in = open(bed_file, 'r')

        for line in bed_in:
            a = line.rstrip().split('\t')
            chrm = a[0]
            start = int(a[1])
            end = int(a[2])

            # extend to pool width
            length = end - start
            if length < pool_width:
                mid = (start + end) // 2
                start = mid - pool_width // 2
                end = start + pool_width

            # add cropped bp
            start = max(0, start - crop_bp)
            end += crop_bp

            # print
            print('%s\t%d\t%d' % (chrm, start, end), file=cat_bed_out)

        bed_in.close()
    cat_bed_out.close()

    # merge
    merge_bed_file = '%s/peaks_merge.bed' % out_dir
    bedtools_cmd = 'bedtools sort -i %s' % cat_bed_file
    bedtools_cmd += ' | bedtools merge -i - > %s' % merge_bed_file
    subprocess.call(bedtools_cmd, shell=True)

    # round and add crop_bp
    full_bed_file = '%s/peaks_full.bed' % out_dir
    full_bed_out = open(full_bed_file, 'w')

    for line in open(merge_bed_file):
        a = line.rstrip().split('\t')
        chrm = a[0]
        start = int(a[1])
        end = int(a[2])
        mid = (start + end) // 2
        length = end - start

        # round length to nearest pool_width
        bins = int(np.round(length / pool_width))
        assert (bins > 0)
        start = mid - (bins * pool_width) // 2
        start = max(0, start)
        end = start + (bins * pool_width)

        # add cropped bp
        # start = max(0, start-crop_bp)
        # end += crop_bp

        # write
        print('%s\t%d\t%d' % (chrm, start, end), file=full_bed_out)

    full_bed_out.close()

    return full_bed_file


################################################################################
def divide_contigs_chr(contigs, test_chrs, valid_chrs):
    """Divide list of contigs into train/valid/test lists
       by chromosome."""
    print('divided by chr')
    # initialize current train/valid/test nucleotides
    train_nt = 0
    valid_nt = 0
    test_nt = 0

    # initialize train/valid/test contig lists
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    # process contigs
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start

        if ctg.chr in test_chrs:
            test_contigs.append(ctg)
            test_nt += ctg_len
        elif ctg.chr in valid_chrs:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        else:
            train_contigs.append(ctg)
            train_nt += ctg_len

    total_nt = train_nt + valid_nt + test_nt

    print('Contigs divided into')
    print(' Train: %5d contigs, %10d nt (%.4f)' % \
          (len(train_contigs), train_nt, train_nt / total_nt))
    print(' Valid: %5d contigs, %10d nt (%.4f)' % \
          (len(valid_contigs), valid_nt, valid_nt / total_nt))
    print(' Test:  %5d contigs, %10d nt (%.4f)' % \
          (len(test_contigs), test_nt, test_nt / total_nt))

    return [train_contigs, valid_contigs, test_contigs]


################################################################################
def divide_contigs_folds(contigs, folds):
    """Divide list of contigs into cross fold lists."""

    # sort contigs descending by length
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    # compute total nucleotides(核苷酸)
    total_nt = sum([lc[0] for lc in length_contigs])

    # compute aimed fold nucleotides
    fold_nt_aim = int(np.ceil(total_nt / folds))

    # initialize current fold nucleotides
    fold_nt = np.zeros(folds)

    # initialize fold contig lists
    fold_contigs = []
    for fi in range(folds):
        fold_contigs.append([])

    # process contigs
    for ctg_len, ctg in length_contigs:
        # compute gap between current and aim
        fold_nt_gap = fold_nt_aim - fold_nt
        fold_nt_gap = np.clip(fold_nt_gap, 0, np.inf)

        # compute sample probability
        fold_prob = fold_nt_gap / fold_nt_gap.sum()

        # sample train/valid/test
        fi = np.random.choice(folds, p=fold_prob)
        fold_contigs[fi].append(ctg)
        fold_nt[fi] += ctg_len

    print('Contigs divided into')
    for fi in range(folds):
        print(' Fold%d: %5d contigs, %10d nt (%.4f)' % \
              (fi, len(fold_contigs[fi]), fold_nt[fi], fold_nt[fi] / total_nt))

    return fold_contigs


################################################################################
def divide_contigs_pct(contigs, test_pct, valid_pct, pct_abstain=0.2):
    """Divide list of contigs into train/valid/test lists, aiming for the specified nucleotide percentages."""
    print('divided by pct')
    # sort contigs descending by length
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    # compute total nucleotides
    total_nt = sum([lc[0] for lc in length_contigs])

    # compute aimed train/valid/test nucleotides
    test_nt_aim = test_pct * total_nt
    valid_nt_aim = valid_pct * total_nt
    train_nt_aim = total_nt - valid_nt_aim - test_nt_aim

    # initialize current train/valid/test nucleotides
    train_nt = 0
    valid_nt = 0
    test_nt = 0

    # initialize train/valid/test contig lists
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    # process contigs
    for ctg_len, ctg in length_contigs:

        # compute gap between current and aim
        test_nt_gap = max(0, test_nt_aim - test_nt)
        valid_nt_gap = max(0, valid_nt_aim - valid_nt)
        train_nt_gap = max(1, train_nt_aim - train_nt)

        # skip if too large
        if ctg_len > pct_abstain * test_nt_gap:
            test_nt_gap = 0
        if ctg_len > pct_abstain * valid_nt_gap:
            valid_nt_gap = 0

        # compute remaining %
        gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
        test_pct_gap = test_nt_gap / gap_sum
        valid_pct_gap = valid_nt_gap / gap_sum
        train_pct_gap = train_nt_gap / gap_sum

        # sample train/valid/test
        ri = np.random.choice(range(3), 1, p=[train_pct_gap, valid_pct_gap, test_pct_gap])[0]
        if ri == 0:
            train_contigs.append(ctg)
            train_nt += ctg_len
        elif ri == 1:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        elif ri == 2:
            test_contigs.append(ctg)
            test_nt += ctg_len
        else:
            print('TVT random number beyond 0,1,2', file=sys.stderr)
            exit(1)

    print('Contigs divided into')
    print(' Train: %5d contigs, %10d nt (%.4f)' % \
          (len(train_contigs), train_nt, train_nt / total_nt))
    print(' Valid: %5d contigs, %10d nt (%.4f)' % \
          (len(valid_contigs), valid_nt, valid_nt / total_nt))
    print(' Test:  %5d contigs, %10d nt (%.4f)' % \
          (len(test_contigs), test_nt, test_nt / total_nt))

    return [train_contigs, valid_contigs, test_contigs]


################################################################################
def limit_contigs(contigs, filter_bed):
    """ Limit to contigs overlapping the given BED.

      Args
       contigs: list of Contigs
       filter_bed: BED file to filter by

      Returns:
       fcontigs: list of Contigs
      """

    # print ctgments to BED
    ctg_fd, ctg_bed_file = tempfile.mkstemp()
    ctg_bed_out = open(ctg_bed_file, 'w')
    for ctg in contigs:
        print('%s\t%d\t%d' % (ctg.chr, ctg.start, ctg.end), file=ctg_bed_out)
    ctg_bed_out.close()

    # intersect w/ filter_bed
    fcontigs = []
    p = subprocess.Popen(
        'bedtools intersect -a %s -b %s' % (ctg_bed_file, filter_bed),
        shell=True,
        stdout=subprocess.PIPE)
    for line in p.stdout:
        a = line.decode('utf-8').split()
        chrom = a[0]
        ctg_start = int(a[1])
        ctg_end = int(a[2])
        fcontigs.append(Contig(chrom, ctg_start, ctg_end))

    p.communicate()

    os.close(ctg_fd)
    os.remove(ctg_bed_file)

    return fcontigs


################################################################################
def rejoin_large_contigs(contigs):
    # 把连着的那些ctg连接成一个长的ctg
    """ Rejoin large contigs that were broken up before alignment comparison."""

    # split list by chromosome
    chr_contigs = {}
    for ctg in contigs:
        chr_contigs.setdefault(ctg.chr, []).append(ctg)

    contigs = []
    for chrm in chr_contigs:
        # sort within chromosome
        chr_contigs[chrm].sort(key=lambda x: x.start)

        ctg_ongoing = chr_contigs[chrm][0]
        for i in range(1, len(chr_contigs[chrm])):
            ctg_this = chr_contigs[chrm][i]
            if ctg_ongoing.end == ctg_this.start:
                # join
                # ctg_ongoing.end = ctg_this.end
                ctg_ongoing = ctg_ongoing._replace(end=ctg_this.end)
            else:
                # conclude ongoing
                contigs.append(ctg_ongoing)

                # move to next
                ctg_ongoing = ctg_this

        # conclude final
        contigs.append(ctg_ongoing)

    return contigs


################################################################################
def write_seqs_bed(bed_file, seqs, labels=False):
    '''Write sequences to BED file.'''
    bed_out = open(bed_file, 'w')
    for i in range(len(seqs)):
        line = '%s\t%d\t%d' % (seqs[i].chr, seqs[i].start, seqs[i].end)
        if labels:
            line += '\t%s' % seqs[i].label
        print(line, file=bed_out)
    bed_out.close()


################################################################################
Contig = collections.namedtuple('Contig', ['chr', 'start', 'end'])
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])
ModelSeq_2 = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label', 'protype'])

################################################################################
if __name__ == '__main__':
    main()
