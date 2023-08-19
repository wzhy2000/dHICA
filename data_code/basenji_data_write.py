#!/usr/bin/env python
from optparse import OptionParser
import os
import sys

import h5py
import numpy as np
import pdb
import pysam

import basenji_data_read as br
from basenji_data import ModelSeq_2
from dna_io import dna_1hot, dna_1hot_index

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
"""
basenji_data_write.py

Write TF Records for batches of model sequences.

Notes:
-I think target_start and target_end are remnants of my previous data2 pipeline.
 If I see this again beyond 8/2020, remove it.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <tfr_file>'
  parser = OptionParser(usage)
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_clip', dest='umap_clip',
      default=1, type='float',
      help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
  parser.add_option('--umap_tfr', dest='umap_tfr',
      default=False, action='store_true',
      help='Save umap array into TFRecords [Default: %default]')
  parser.add_option('-x', dest='extend_bp',
      default=0, type='int',
      help='Extend sequences on each side [Default: %default]')
  parser.add_option('--zzx', dest='zzx',
      default=False, action='store_true',
      help='zzx') #没有用上
  parser.add_option('--mouse', dest='mouse',
      default=False, action='store_true',
      help='Is mouse?')
  (options, args) = parser.parse_args()

  if len(args) != 5:
    parser.error('Must provide input arguments.')
  else:
    fasta_file = args[0]
    seqs_bed_file = args[1]
    seqs_cov_dir = args[2]
    proseq_id_file = args[3] # id.txt
    tfr_file = args[4] #tfr_stem (tfr_dir, fold_set, tfr_i)

  # exit()
  ################################################################
  # read model sequences

  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq_2(a[0],int(a[1]),int(a[2]), a[3], a[4])) #['chr', 'start', 'end', 'label', 'protype']

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i

  ################################################################
  # determine sequence coverage files

  seqs_cov_files = []
  ti = 0
  seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)
  while os.path.isfile(seqs_cov_file):
    seqs_cov_files.append(seqs_cov_file)
    ti += 1
    seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)

  if len(seqs_cov_files) == 0:  
    print('Sequence coverage files not found, e.g. %s' % seqs_cov_file, file=sys.stderr)
    exit(1)

  #print(seqs_cov_files[0])
  seq_pool_len = h5py.File(seqs_cov_files[0], 'r')['targets'].shape[1]
  num_targets = len(seqs_cov_files)

  ################################################################
  # read targets

  # initialize targets
  # targets = np.zeros((num_seqs, 1028, num_targets), dtype='float16')
  # 原来的
  # seq_pool_len是8192
  targets = np.zeros((num_seqs, seq_pool_len, num_targets), dtype='float16')

  # read each target
  for ti in range(num_targets):
    seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
    # print(seqs_cov_open)
    targets[:,:,ti] = seqs_cov_open['targets'][options.start_i:options.end_i,:]
    seqs_cov_open.close()

  ################################################################
  # modify unmappable

  if options.umap_npy is not None and options.umap_clip < 1:
    unmap_mask = np.load(options.umap_npy)

    for si in range(num_seqs):
      msi = options.start_i + si

      # determine unmappable null value
      seq_target_null = np.percentile(targets[si], q=[100*options.umap_clip], axis=0)[0]

      # set unmappable positions to null
      targets[si,unmap_mask[msi,:],:] = np.minimum(targets[si,unmap_mask[msi,:],:], seq_target_null)

  elif options.umap_npy is not None and options.umap_tfr:
    unmap_mask = np.load(options.umap_npy)

  ################################################################
  # write TFRecords

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)

  # define options
  tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

  with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
    for si in range(num_seqs):
      msi = options.start_i + si
      mseq = model_seqs[msi]
      mseq_start = mseq.start - options.extend_bp
      mseq_end = mseq.end + options.extend_bp

      # read FASTA
      # seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)
      # 想拓展，得在这里入手
      '''
      fetch里已经有补N了，这里不用补了
      '''
      # if options.zzx:
      #   # 判断是否越界，越界补N
      #   mseq_start = mseq_start - 32512
      #   mseq_end = mseq_end + 32512
      #   if mseq_start < 0 or mseq_end > chr_length[mseq.chr]:
      #     fasta_start = max(mseq_start, 0)
      #     fasta_end = min(chr_length[mseq.chr], mseq_end)
      #     pad_upstream = 'N' * max(mseq_start, 0)
      #     pad_downstream = 'N' * max(mseq_end - chr_length[mseq.chr], 0)
      #     seq_dna = fetch_dna(fasta_open, mseq.chr, fasta_start, fasta_end)
      #     seq_dna = pad_upstream + seq_dna + pad_downstream
      #   else:
      #     seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)
      # else:
      #   seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)
      # print(mseq_start, mseq_end)
      mseq_start = mseq_start - 32768
      mseq_end = mseq_end + 32768
      seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)#处理超出track范围的sequence
      # print(tfr_file, si, mseq.chr, len(seq_dna), mseq.start, mseq.end)
      # print(seq_dna[:10])
      # one hot code (N's as zero)
      seq_1hot = dna_1hot(seq_dna, n_uniform=False, n_sample=False)
   
      # seq_1hot = dna_1hot_index(seq_dna) # more efficient, but fighting inertia
      # if np.all(seq_1hot==0) or np.all(targets[si,:,:]==0):
      #   with open(seqs_bed_file[:-3]+'miss.bed', 'a') as w_obj:
      #     w_obj.write(mseq.chr + '\t' + str(mseq.start) + '\t' +  str(mseq.end) + '\t' + mseq.label + '\t' +  mseq.protype + '\n')
      #   continue

      # print(type(seq_1hot))
      # exit()
      # read pro-seq
      # proseq_id_file = '/local/hg19_data/proseq_id.txt'
      # print(mseq_start, mseq_end)
      #pro_seq_minus, pro_seq_plus, pro_seq_minus_plus = np.asarray(get_pro_seq(proseq_id_file, mseq.chr, mseq_start, mseq_end, mseq.protype))
      atac_seq = np.asarray(get_atac_seq(proseq_id_file, mseq.chr, mseq_start, mseq_end, mseq.protype)) #根据不同的protype读取不同的bw文件

      # print(pro_seq[0].shape)
      # exit()
      
      # print(len(pro_seq))
      # exit()

      # hash to bytes
      

      atac_seq = abs(atac_seq)

      targets_n = targets[si,:,:]
      atac_seq[np.where(np.isnan(atac_seq))] = 1e-3

      targets_n[np.where(np.isnan(targets_n))] = 1e-3
      atac_seq[np.where(np.isinf(atac_seq))] = 1e-3

      targets_n[np.where(np.isinf(targets_n))] = 1e-3


      if mseq.chr == 'chrX':
        start_end = [23, mseq_start, mseq_end]
      else:
        start_end = [mseq.chr[3:], mseq_start, mseq_end]

      start_end = np.asarray(start_end).astype('int32')

      features_dict = {
        'sequence': feature_bytes(seq_1hot),
        'atac-seq': feature_bytes(atac_seq),
        'start-end': feature_bytes(start_end),
        'target': feature_bytes(targets_n)
        }

      # add unmappability
      if options.umap_tfr:
        features_dict['umap'] = feature_bytes(unmap_mask[msi,:])

      # write example
      example = tf.train.Example(features=tf.train.Features(feature=features_dict))
      writer.write(example.SerializeToString())

    fasta_open.close()

'''
  genome_cov_file = '/local/hg19_data/pro-seq/GSM1480325_K562_GROseq_minus.bigWig'
  genome_cov_open = br.CovFace(genome_cov_file)
  seq_cov_nt = genome_cov_open.read(mseq.chr, mseq_start, mseq_end)
  baseline_cov = np.percentile(seq_cov_nt, 100*0.5)
  baseline_cov = np.nan_to_num(baseline_cov)
  nan_mask = np.isnan(seq_cov_nt)
  seq_cov_nt[nan_mask] = baseline_cov
  pro_seq = seq_cov_nt
'''

def get_atac_seq(atac_id_file, chr, start, end, protype):
  atac_seq = []
  # with open(proseq_id_file, 'r') as r_obj:
  #   genome_cov_files = r_obj.readlines()
  # for genome_cov_file in genome_cov_files:
  # genome_cov_file = genome_cov_file[:-1]

  genome_cov_file = '/local/ww/enformer/enformer_data/epigenomes/bigWig/' + protype + '-DNase.fc.signal.bigwig'

  # genome_cov_file_plus = '/local/hg19_data/proseq/' + protype + '_plus.bw'
  try:
    genome_cov_open = br.CovFace(genome_cov_file)
    #genome_cov_open_plus = br.CovFace(genome_cov_file_plus)
  except:
    print('111', protype)
    exit()

  p_start = start if start > 0 else 0
  p_end = end if end < chr_length_human[chr] else chr_length_human[chr]

  try:
    seq_cov_nt = genome_cov_open.read(chr, p_start, p_end)
    #seq_cov_nt_plus = genome_cov_open_plus.read(chr, p_start, p_end)
  except:
    print(chr, start, end)
    print(chr, p_start, p_end)
    exit()

  baseline_cov = np.percentile(seq_cov_nt, 100 * 0.5)
  #baseline_cov_plus = np.percentile(seq_cov_nt_plus, 100 * 0.5)

  baseline_cov = np.nan_to_num(baseline_cov)
  #baseline_cov_plus = np.nan_to_num(baseline_cov_plus)

  nan_mask = np.isnan(seq_cov_nt)
  #nan_mask_plus = np.isnan(seq_cov_nt_plus)

  seq_cov_nt[nan_mask] = baseline_cov
  #seq_cov_nt_plus[nan_mask_plus] = baseline_cov_plus

  # 拼接越界部分，越界部分赋值为0
  seq_cov_nt = np.hstack((np.zeros(abs(start - p_start)), seq_cov_nt))
  seq_cov_nt = np.hstack((seq_cov_nt, np.zeros(abs(end - p_end)))).astype('float16')

  #seq_cov_nt_plus = np.hstack((np.zeros(abs(start - p_start)), seq_cov_nt_plus))
  #seq_cov_nt_plus = np.hstack((seq_cov_nt_plus, np.zeros(abs(end - p_end)))).astype('float16')

  #proseq_minus_plus = abs(seq_cov_nt_minus) + abs(seq_cov_nt_plus)

  atac_seq.append(seq_cov_nt)
  #pro_seq_plus.append(seq_cov_nt_plus)
  #pro_seq_minus_plus.append(proseq_minus_plus)

  '''
  seq_cov_nt = seq_cov_nt_minus + seq_cov_nt_plus

  # 拼接越界部分，越界部分赋值为0
  seq_cov_nt = np.hstack((np.zeros(abs(start-p_start)), seq_cov_nt))
  seq_cov_nt = np.hstack((seq_cov_nt, np.zeros(abs(end-p_end))))
  #
  # 改东西时需要调整的参数
  #
  seq_cov_nt = seq_cov_nt.reshape(1536, 128)
  seq_cov_nt = seq_cov_nt.sum(axis=1, dtype='float16')

  pro_seq.append(seq_cov_nt)
  '''

  return atac_seq

def get_atac_seq1(atac_id_file, chr, start, end, protype):
  atac_seq = []
  with open(atac_id_file, 'r') as r_obj:
    genome_cov_files = r_obj.readlines()
  genome_cov_files = [line.strip("\n") for line in genome_cov_files]

  for genome_cov_file in genome_cov_files:
    genome_cov_file = genome_cov_file[:]
    try:
      genome_cov_open = br.CovFace(genome_cov_file)
    except:
      print('111', protype)
      exit()

    p_start = start if start > 0 else 0
    p_end = end if end < chr_length_human[chr] else chr_length_human[chr]

    try:
      seq_cov_nt = genome_cov_open.read(chr, p_start, p_end)
      # seq_cov_nt_plus = genome_cov_open_plus.read(chr, p_start, p_end)
    except:
      print(chr, start, end)
      print(chr, p_start, p_end)
      exit()

    baseline_cov = np.percentile(seq_cov_nt, 100 * 0.5)
    # baseline_cov_plus = np.percentile(seq_cov_nt_plus, 100 * 0.5)

    baseline_cov = np.nan_to_num(baseline_cov)
    # baseline_cov_plus = np.nan_to_num(baseline_cov_plus)

    nan_mask = np.isnan(seq_cov_nt)

    seq_cov_nt[nan_mask] = baseline_cov

    # 拼接越界部分，越界部分赋值为0
    seq_cov_nt = np.hstack((np.zeros(abs(start - p_start)), seq_cov_nt))
    seq_cov_nt = np.hstack((seq_cov_nt, np.zeros(abs(end - p_end)))).astype('float16')

    atac_seq.append(seq_cov_nt)

  return atac_seq



def fetch_dna(fasta_open, chrm, start, end):
  """Fetch DNA when start/end may reach beyond chromosomes."""

  # initialize sequence
  seq_len = end - start
  seq_dna = ''

  # add N's for left over reach
  if start < 0:
    seq_dna = 'N'*(-start)
    start = 0

  # get dna
  seq_dna += fasta_open.fetch(chrm, start, end)

  # add N's for right over reach
  if len(seq_dna) < seq_len:
    seq_dna += 'N'*(seq_len-len(seq_dna))

  return seq_dna


def feature_bytes(values):
  """Convert numpy arrays to bytes features."""
  values = values.flatten().tostring()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def feature_floats(values):
  """Convert numpy arrays to floats features.
     Requires more space than bytes."""
  values = values.flatten().tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def normali(Z):
  norm = np.linalg.norm(Z, axis=0)
  out = Z/norm
  return out

# def normali_column(Z):
#   norm = np.linalg.norm(Z, axis=0)
#   out = Z/norm
#   return out

def normali_column(Z):
  Zmax,Zmin=Z.max(axis=1),Z.min(axis=1)
  for i in range(len(Z)):
    if (Zmax[i]-Zmin[i]) == 0:
      Z[i] = np.full(len(Z[i]), 1e-3)
    else:
      Z[i]=(Z[i]-Zmin[i])/(Zmax[i]-Zmin[i])
  return Z


chr_length_human = {'chr1':249250621, 'chr2':243199373, 'chr3':198022430, 'chr4':191154276, 'chr5':180915260, 'chr6':171115067, 'chr7':159138663, 'chr8': 146364022, 'chr9':141213431, 'chr10':135534747, 
'chr11':135006516, 'chr12':133851895, 'chr13':115169878, 'chr14':107349540, 'chr15':102531392, 'chr16':90354753, 'chr17':81195210, 'chr18':78077248, 'chr19':59128983, 'chr20':63025520,
'chr21':48129895, 'chr22':51304566, 'chrX':155270560, 'chrY':59373566}
chr_length_mouse = {'chr1':195471971, 'chr2':242193529, 'chr3':160039680, 'chr4':156508116, 'chr5':151834684, 'chr6':149736546, 'chr7':145441459, 'chr8': 129401213, 'chr9':124595110, 'chr10':130694993,
'chr11':122082543, 'chr12':120129022, 'chr13':120421639, 'chr14':124902244, 'chr15':104043685, 'chr16':98207768, 'chr17':94987271, 'chr18':90702639, 'chr19':61431566,
'chrX':171031299, 'chrY':91744698}
################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
