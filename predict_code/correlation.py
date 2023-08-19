from optparse import OptionParser
import os
from traceback import print_tb
import pyBigWig
import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr, spearmanr


class CovFace:
  def __init__(self, cov_file):
    self.cov_file = cov_file
    self.bigwig = False
    self.bed = False

    cov_ext = os.path.splitext(self.cov_file)[1].lower()
    if cov_ext == '.gz':
      cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

    if cov_ext in ['.bed', '.narrowpeak']:
      self.bed = True
      self.preprocess_bed()

    elif cov_ext in ['.bw','.bigwig']:
      self.cov_open = pyBigWig.open(self.cov_file, 'r')
      self.bigwig = True

  def preprocess_bed(self):
    # read BED
    bed_df = pd.read_csv(self.cov_file, sep='\t',
      usecols=range(3), names=['chr','start','end'])

    # for each chromosome
    self.cov_open = {}
    for chrm in bed_df.chr.unique():
      bed_chr_df = bed_df[bed_df.chr==chrm]

      # find max pos
      pos_max = bed_chr_df.end.max()

      # initialize array
      self.cov_open[chrm] = np.zeros(pos_max, dtype='bool')

      # set peaks
      for peak in bed_chr_df.itertuples():
        self.cov_open[peak.chr][peak.start:peak.end] = 1


  def read(self, chrm, start, end):
    if self.bigwig:
      cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')
    else:
      if chrm in self.cov_open:
        cov = self.cov_open[chrm][start:end]
        pad_zeros = end-start-len(cov)
        if pad_zeros > 0:
          cov_pad = np.zeros(pad_zeros, dtype='bool')
          cov = np.concatenate([cov, cov_pad])
      else:
        print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % \
          (self.cov_file, chrm, start, end), file=sys.stderr)
        cov = np.zeros(end-start, dtype='float16')

    return cov

  def chr_length(self):
      return self.cov_open.chroms()

  def close(self):
    if not self.bed:
      self.cov_open.close()


def get_start_end(path, chr_list):
    """
    Distinguish peaks by chromosome
    Args:
        path: path of peak file
        chr_list: a dict like {chromosome: 'lenth of chromosome'}

    Output:
          chr_peaks: a dict consisted of peaks in different chromosome
    """
    with open(path, 'r') as r_obj:
        sections = r_obj.readlines()
    sections = [section.split()[:3] for section in sections]
    chr_peaks = {}

    for chrome in chr_list:
      chr_peaks[chrome] = []

    for section in sections:
      if section[0] in chr_peaks.keys():
        chr_peaks[section[0]].append({'start': section[1], 'end': section[2]})

    return chr_peaks

def corr_peak(path_peak, genome_cov_open_predicted, genome_cov_open_exper, chr_list):
  """
  Calculate the mean of gene expression for each peak fragment, and then calculate the correlation across the genome.

  Args:
        path_peak: path of peak file
        genome_cov_open_a: CovFace object of a.bw
        genome_cov_open_b: CovFace object of b.bw
        chr_list: a dict like {chromosome: 'lenth of chromosome'}

  Output:
        (float) pearsonr correlation
        (float) spearman correlation
  """

  peaks = get_start_end(path_peak, chr_list)
  pre_all = []
  exper_all = []
  for chr in peaks:
      chr_peaks = peaks[chr]
      for peak in chr_peaks:
          start = int(peak['start'])
          end =  int(peak['end'])
          try:
              predicted = genome_cov_open_predicted.read(chr, start, end)
          except:
              continue
          predicted[np.where(np.isnan(predicted))] = 1e-3
          
          exper = genome_cov_open_exper.read(chr, start, end)
          exper[np.where(np.isnan(exper))] = 1e-3

          if (predicted == predicted[0]).all() or (exper == exper[0]).all():
              continue
          predicted = np.mean(predicted)
          exper = np.mean(exper)

          pre_all.append(predicted)
          exper_all.append(exper)
            
  pre_all = np.asarray(pre_all, dtype='float32')
  exper_all = np.asarray(exper_all, dtype='float32')
  correlation_pearsonr = round(pearsonr(pre_all, exper_all)[0], 4)
  correlation_spearmanr = round(spearmanr(pre_all, exper_all)[0], 4)
  return correlation_pearsonr, correlation_spearmanr

def corr_resolution(genome_cov_open_a, genome_cov_open_b, chr_list, resolution):
  """
  Calculate the correlation between a.bw and b.bw in terms of "resolution"
  Divide the genome into different small fragments according to the resolution, 
  calculate the average gene expression level of each small fragment, 
  and then calculate the correlation of the whole genome.

  Args:
        genome_cov_open_a: CovFace object of a.bw
        genome_cov_open_b: CovFace object of b.bw
        chr_list: a dict like {chromosome: 'lenth of chromosome'}
        resolutiin: the resolution user specified

  Output:
        (float) pearsonr correlation
        (float) spearman correlation
  """
  a = []
  b = []
  for chr in chr_list:
    chr_length = chr_list[chr]

    end = chr_length - chr_length%resolution

    chr_data_a = genome_cov_open_a.read(chr, 0, end)
    chr_data_b = genome_cov_open_b.read(chr, 0, end)
    chr_data_a[np.where(np.isnan(chr_data_a))] = 1e-3
    chr_data_b[np.where(np.isnan(chr_data_b))] = 1e-3

    chr_data_a = chr_data_a.reshape(-1, resolution)
    chr_data_a = np.mean(chr_data_a, axis=-1)

    chr_data_b = chr_data_b.reshape(-1, resolution)
    chr_data_b = np.mean(chr_data_b, axis=-1)

    a = np.hstack((a, chr_data_a))
    b = np.hstack((b, chr_data_b))

  a = np.asarray(a, dtype='float32')
  b = np.asarray(b, dtype='float32')
  correlation_pearsonr = round(pearsonr(a, b)[0], 4)
  correlation_spearmanr = round(spearmanr(a, b)[0], 4)
  return correlation_pearsonr, correlation_spearmanr


def correlation_call(bigwig_file_a, bigwig_file_b, peak_resolution, chr=None):
  path_a = bigwig_file_a
  path_b = bigwig_file_b

  # read a.bw b.bw
  genome_cov_open_a = CovFace(path_a)
  genome_cov_open_b = CovFace(path_b)

  chr_list = genome_cov_open_a.chr_length()

  # specify one chromosome
  if chr:
    chr_list = {chr: chr_list[chr]}

  # determine whether the input is a path or a resolution
  try:
    int(peak_resolution)
    peak_file = False
  except:
    peak_file = peak_resolution
  
  # calculate the correlation between a.bw and b.bw
  if peak_file:
    correlation_pearsonr, correlation_spearmanr = corr_peak(peak_file, genome_cov_open_a, genome_cov_open_b, chr_list)
    print('Pearson correlation:', correlation_pearsonr)
    print('Spearman correlation', correlation_spearmanr)
  else:
    resolution = int(peak_resolution)
    correlation_pearsonr, correlation_spearmanr = corr_resolution(genome_cov_open_a, genome_cov_open_b, chr_list, resolution)
    print('Pearson correlation:', correlation_pearsonr)
    print('Spearman correlation', correlation_spearmanr)


def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file>'
    parser = OptionParser(usage)

    parser.add_option('-a', dest='bigwig_file_a',
      default='bw_a',
      help='BigWig file a [Default: %default]')

    parser.add_option('-b', dest='bigwig_file_b',
      default='bw_b',
      help='BigWig file b [Default: %default]')

    parser.add_option('-p', dest='peak_resolution',
      default='p_r',
      help='peak file or resolution(100, 1000, 10000, ...) [Default: %default]')

    parser.add_option('--chr', dest='chr',
      help='specify the chromosome [Default: %default]')

    (options, args) = parser.parse_args()

    if (options.bigwig_file_a == 'bw_a' or options.bigwig_file_b == 'bw_b' or options.peak_resolution == 'p_r'):
      print('Useage: python correlation.py -a <a.bw> -b <b.bw> -p <path or resolution(100, 1000,...)>')
      print('Options:')
      print('\t--chr - Only calculate the correlation between a.bw and b.bw on a certain chromosome. ' + 
        'If this parameter is not selected, the correlation between the two files on all chromosomes is calculated by default.et.')
      print('Example: python correlation.py -a a.bw -b b.bw -p 10000 --chr 22')
      exit()

    path_a = options.bigwig_file_a
    path_b = options.bigwig_file_b

    # read a.bw b.bw
    genome_cov_open_a = CovFace(path_a)
    genome_cov_open_b = CovFace(path_b)

    chr_list = genome_cov_open_a.chr_length()

    # specify one chromosome
    if options.chr:
      chr_list = {options.chr: chr_list[options.chr]}

    # determine whether the input is a path or a resolution
    try:
      int(options.peak_resolution)
      peak_file = False
    except:
      peak_file = options.peak_resolution

    # calculate the correlation between a.bw and b.bw 
    if peak_file:
      correlation_pearsonr, correlation_spearmanr = corr_peak(peak_file, genome_cov_open_a, genome_cov_open_b, chr_list)
      print('Pearson correlation:', correlation_pearsonr)
      print('Spearman correlation', correlation_spearmanr)
    else:
      resolution = int(options.peak_resolution)
      correlation_pearsonr, correlation_spearmanr = corr_resolution(genome_cov_open_a, genome_cov_open_b, chr_list, resolution)
      print(path_b)
      print(correlation_pearsonr, '\t', correlation_spearmanr)
      # print('Spearman correlation', correlation_spearmanr)




if __name__ == '__main__':
    main()
