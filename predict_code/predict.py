from optparse import OptionParser
import subprocess
import os
import random
import pyBigWig
from tensorflow.python.keras.backend import print_tensor
from Roformer_data import genome



def roformer_call(out_dir, atac, model_path, ref_genome=None, processes=8, multi=False):
  # get the working path and the path where the program is located
  work_path = os.getcwd()
  current_path = os.path.dirname(__file__)

  ################################################################################
  # make dataset
  ################################################################################
  roformer_data_path = os.path.join(current_path, 'Roformer_data.py')

   # make directory of dataset
  random_id = random.randint(0, 99999)
  dataset_outpath = os.path.join(work_path, str(random_id))

  # if the dataset folder already exists, re-select a random number
  while os.path.isdir(dataset_outpath):
      random_id = random.randint(0, 99999)
      dataset_outpath = os.path.join(work_path, str(random_id))

  # user provid reference genome file
  if not(ref_genome):
    cmd_data_make = ['python', roformer_data_path, '--local','-o', dataset_outpath, '-p', str(processes), 
        '--ref', 'None', atac]
  # user does not provide reference genome file
  else:
    cmd_data_make = ['python', roformer_data_path, '--local','-o', dataset_outpath, '-p', str(processes), 
      '--ref', ref_genome, atac]
  
  # call
  subprocess.call(cmd_data_make)
  print('Dataset has down')

  ################################################################################
  # predict
  ################################################################################
  # use multiple GPUs
  if multi:
    predict_code_path = os.path.join(current_path, 'predicted.py')
  # use one GPU
  else:
    predict_code_path = os.path.join(current_path, 'one_predicted.py')

  # make directory of result
  if not(os.path.isdir(out_dir)):
        os.mkdir(out_dir)
  output = os.path.join(out_dir, str(random_id))
  if not(os.path.isdir(output)):
      os.mkdir(output)

  # choose model type
  if not(ref_genome):
    model_types = ['R']
  else:
    model_types = ['R', 'D', 'RD']

  # predict
  for model_type in model_types:
      cmd_predicted = ['python', predict_code_path, '-d', dataset_outpath, '--model_type', model_type, '--idx', os.path.join(dataset_outpath, 'chr_length.bed'), 
          '--idx_chr', os.path.join(dataset_outpath, 'contigs.bed'), '-o', os.path.join(output, model_type), '-m', model_path]
      subprocess.call(cmd_predicted)

  # delete the file of dataset
  cmd_rm = ['rm', '-rf', dataset_outpath]
  subprocess.call(cmd_rm)


def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir',
      default='result_out',
      help='output directory of predicted result [Default: %default]')
    parser.add_option('-m', dest='model_path',
      default='model_path',
      help='model Path(contain R, D, RD models) [Default: %default]')
    parser.add_option('-p', dest='processes',
      default=8, type='int',
      help='number parallel processes [Default: %default]')
    parser.add_option('--ref', dest='ref_genome',
      help='reference genome(*.fasta) [Default: %default]')
    parser.add_option('--atac', dest='atac',
      default='atac',
      help='atac-seq bigWig file [Default: %default]')
    parser.add_option('--multi', dest='multi',
      default=False, action='store_true',
      help='use muiti-GPU?. [Default: %default]')
    parser.add_option('--op', dest='output_pre',
      default='out',
      help='Prefix for output file [Default: %default]')

    (options, args) = parser.parse_args()

    if (options.out_dir == 'result_out' or options.model_path == 'model_path' or options.atac == 'atac'):
      print('Useage: python3 predict.py -o <output_path> -m <model_path> --atac <atac-seq.bw>')
      print('Requirements: <sort-bed>, <bedGraphToBigWig>')
      print('Options:')
      print('\t-p - The number of processes used to make the dataset')
      print('\t--ref - The reference genome file <genome.fasta>')
      print('\t--multi - Whether to use multiple GPUs? The default value is False. If you select this parameter, multiple GPU will be used')
      print('\t--op Prefix for output file')
      exit()

    # get the working path and the path where the program is located
    work_path = os.getcwd()
    current_path = os.path.dirname(__file__)

    ################################################################################
    # make dataset
    ################################################################################
    roformer_data_path = os.path.join(current_path, 'Roformer_data.py')
    
    # make directory of dataset
    random_id = random.randint(0, 99999)
    dataset_outpath = os.path.join(work_path, str(random_id))

    # if the dataset folder already exists, re-select a random number
    while os.path.isdir(dataset_outpath):
        random_id = random.randint(0, 99999)
        dataset_outpath = os.path.join(work_path, str(random_id))

    # user provid reference genome file
    if not(options.ref_genome):
      cmd_data_make = ['python', roformer_data_path, '--local','-o', dataset_outpath, '-p', str(options.processes), 
          '--ref', 'None', options.atac]
    # user does not provide reference genome file
    else:
      fasta_length = genome.load_chromosomes(options.ref_genome)

      bw = pyBigWig.open(options.atac)
      bw_length = bw.chroms()

      for i in range(1, 21):
        if fasta_length['chr%d'%i][0][1] != bw_length['chr%d'%i]:
          print('################################################################################')
          print('Please provide the correct reference genome file.')
          print('################################################################################')
          exit()

      cmd_data_make = ['python', roformer_data_path, '--local','-o', dataset_outpath, '-p', str(options.processes), 
        '--ref', options.ref_genome, options.atac]
    
    # call
    subprocess.call(cmd_data_make)
    print('Dataset has down')

    ################################################################################
    # predict
    ################################################################################
    # use multiple GPUs
    if options.multi:
      predict_code_path = os.path.join(current_path, 'predicted.py')
    # use one GPU
    else:
      predict_code_path = os.path.join(current_path, 'one_predicted.py')
    
    # make directory of result
    if not(os.path.isdir(options.out_dir)):
      os.mkdir(options.out_dir)
    output = os.path.join(options.out_dir, 'predicted_bw')
    if not(os.path.isdir(output)):
      os.mkdir(output)

    # choose model type
    if not(options.ref_genome):
      model_types = ['R']
    else:
      model_types = ['RD', 'R', 'D']

    # predict
    for model_type in model_types:
        cmd_predicted = ['python', predict_code_path, '-d', dataset_outpath, '--model_type', model_type, '--idx', os.path.join(dataset_outpath, 'chr_length.bed'), 
            '--idx_chr', os.path.join(dataset_outpath, 'contigs.bed'), '-o', output, '--op', options.output_pre, '-m', options.model_path]
        print(cmd_predicted)
        subprocess.call(cmd_predicted)

    print('The result is in the %s' % output)

    ## delete the file of dataset
    cmd_rm = ['rm', '-rf', dataset_outpath]
    subprocess.call(cmd_rm)




if __name__ == '__main__':
    main()


