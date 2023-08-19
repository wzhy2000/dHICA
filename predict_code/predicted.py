import random
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import json
import functools
from model_enfo import Enformer as Enforme_R
from model_enfo import Enformer as Enforme_D
from model_enfo import Enformer as Enforme_RD
import time
from operator import itemgetter
import subprocess
from optparse import Option, OptionParser

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def get_metadata(data_path):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(data_path, 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def tfrecord_files(data_path, subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        data_path, 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac-seq': tf.io.FixedLenFeature([], tf.string),
        'start-end': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_example(serialized_example, feature_map)

    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    atac = tf.io.decode_raw(example['atac-seq'], tf.float16)
    atac = tf.reshape(atac, (metadata['atac_length'], metadata['num_atacseq']))
    atac = tf.cast(atac, tf.float32)

    start_end = tf.io.decode_raw(example['start-end'], tf.int32)
    start_end = tf.reshape(start_end, (1, 3))
    start_end = tf.cast(start_end, tf.int32)

    return {
        'sequence': sequence,
        'atac-seq': atac,
        'start-end': start_end
    }

def get_dataset(data_path, subset, num_threads=8):
    metadata = get_metadata(data_path)
    dataset = tf.data.TFRecordDataset(tfrecord_files(data_path, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads)
    # print(tfrecord_files(organism, subset))
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                          num_parallel_calls=num_threads)
    return dataset


def write_bedGraph(results, out_path, chr_length_human, pre_out, is_chr22=False):
    for chr in results:
        chr_result = results[chr]
        chr_result = sorted(chr_result, key=itemgetter('start'))

        for j in range(len(histone_list)):
            with open(os.path.join(out_path, pre_out + '-' + histone_list[j] + '.bedgraph'), 'a') as w_obj:
                if is_chr22:
                    if chr == 'chr22':
                        if chr_result[0]['start'] > 0:
                            w_obj.write(chr + '\t' + str(0) + '\t' + str(chr_result[0]['start']) + '\t' + str(0) + '\n')
                    for item in chr_result:
                        for i in range(896):
                            start = item['start'] + i * 128
                            end = start + 128
                            w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                    if chr == 'chr22':
                        if chr_result[-1]['end'] < chr_length_human[chr]:
                            w_obj.write(chr + '\t' + str(chr_result[-1]['end']) + '\t' + str(chr_length_human[chr]) + '\t' + str(0) + '\n')
                else:
                    try:
                        if chr_result[0]['start'] > 0:
                            w_obj.write(chr + '\t' + str(0) + '\t' + str(chr_result[0]['start']) + '\t' + str(0) + '\n')
                    except:
                        print(chr_result)
                        print(chr)

                    last_end = 0
                    for item in chr_result:
                        if item['start'] >= last_end:
                            for i in range(896):
                                start = item['start'] + i * 128
                                end = start + 128
                                w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                        
                        else:
                            gap_h = last_end - item['start']
                            h_start = gap_h // 128
                            w_obj.write(chr + '\t' + str(last_end) + '\t' + str(item['start'] + 128 * (h_start+1)) + '\t' + str(item['predicted'][h_start][j]) + '\n')
                            for i in range(h_start+1, 896):
                                start = item['start'] + i * 128
                                end = start + 128
                                w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                        last_end = item['end']
                        

                    try:
                        if chr_result[-1]['end'] < chr_length_human[chr]:
                            w_obj.write(chr + '\t' + str(chr_result[-1]['end']) + '\t' + str(chr_length_human[chr]) + '\t' + str(0) + '\n')
                    except:
                        print('an error', chr)
                        


def evaluate_model(model, dataset, chr_length_human, ID_to_chr_dict, max_steps=None):

    def predict(batch):
        return model(batch['sequence'], batch['atac-seq'], is_training=False)['human']

    @tf.function
    def distributed_predict_step(dist_inputs):
        predicted = mirrored_strategy.run(predict, args=(dist_inputs,))
        return predicted
    # len_dataset = 0

    results = {}
    for chr in chr_length_human:
        results[chr] = []

    for i, batch in tqdm(enumerate(dataset)):
        if max_steps is not None and i > max_steps:
            break
        predicted = distributed_predict_step(batch)

        predicted = predicted.values
        start_end = batch['start-end'].values
        for i in range(len(predicted)):
            result = {}

            try:
                chr_id = start_end[i][0][0].numpy()[0]
            except:
                continue

            chr = ID_to_chr_dict[str(chr_id)]
            result['start'] = start_end[i][0][0].numpy()[1] + 40960
            result['end'] = start_end[i][0][0].numpy()[2] - 40960
            result['predicted'] = predicted[i][0].numpy()
            results[chr].append(result)

    return results


mirrored_strategy = tf.distribute.MirroredStrategy()

def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='dataset',
      default='data_set',
      help='Dataset path [Default: %default]')
    parser.add_option('--model_type', dest='modelType',
      default='R', type='str',
      help='model you choosed R:roseq-only, D:DNA-only, RD:roseq+DNA [Default: %default]')
    parser.add_option('--idx', dest='index',
      default='idx_set',
      help='index file of bedGrapgToBigwig [Default: %default]')
    parser.add_option('--idx_chr', dest='indexChr',
      default='idx_set',
      help='index file of chromosome [Default: %default]')
    parser.add_option('-o', dest='outpath',
      default='outpath',
      help='Output path [Default: %default]')
    parser.add_option('-m', dest='model_path',
      default='model_path',
      help='Model Path(contain R, D, RD models) [Default: %default]')
    parser.add_option('--op', dest='output_pre',
      default='out',
      help='Prefix for output file [Default: %default]')

    (options, args) = parser.parse_args()

    # chr dict
    chr_length_file = options.indexChr
    chr_length_human = make_length_dict(chr_length_file)
    ID_to_chr_dict = make_chr_id(chr_length_file)

    # make output dir
    out_path = options.outpath
    if not(os.path.isdir(out_path)):
        os.mkdir(out_path)

    # set batchsize
    batch_size_replica = 1
    global_batch_size = batch_size_replica * mirrored_strategy.num_replicas_in_sync

    # choose model
    with mirrored_strategy.scope():
        if options.modelType == 'R':
            model = Enforme_R(channels=768,
                             num_heads=8,
                             num_transformer_layers=11,
                             pooling_type='max')
            weight_path = os.path.join(options.model_path, 'model.ckpt')
            model.load_weights(weight_path)
        elif options.modelType == 'D':
            model = Enforme_D(channels=768,
                             num_heads=8,
                             num_transformer_layers=11,
                             pooling_type='max')
            weight_path = os.path.join(options.model_path, 'model.ckpt')
            model.load_weights(weight_path)
        else:
            model = Enforme_RD(channels=768,
                             num_heads=8,
                             num_transformer_layers=11,
                             pooling_type='max')
            weight_path = os.path.join(options.model_path, 'model.ckpt')
            model.load_weights(weight_path)

    # load dataset
    dataset = get_dataset(options.dataset, 'train').batch(global_batch_size).prefetch(2)
    distributed_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

    # predict
    results = evaluate_model(model,
                            distributed_dataset,
                            chr_length_human=chr_length_human,
                            ID_to_chr_dict=ID_to_chr_dict,
                            max_steps=100000)

    pre_out = options.output_pre + '-' + options.modelType

    # write results to file.bedgraph
    write_bedGraph(results, out_path, chr_length_human, pre_out, is_chr22=False)
    
    # .bedgraph to .bw
    for histone in histone_list:
        bedgraph_path = os.path.join(out_path, pre_out + '-' + histone + '.bedgraph')

        bedgraph_path_sorted = os.path.join(out_path, pre_out + '-' + histone + '_sorted.bedgraph')
        bw_path = os.path.join(out_path, pre_out + '-' + histone + '.bw')
        hg19_idx = options.index
        cmd_bedSort = 'sort-bed ' + bedgraph_path + ' > ' + bedgraph_path_sorted
        p = subprocess.Popen(cmd_bedSort, shell=True)
        p.wait()

        cmd = ['bedGraphToBigWig', bedgraph_path_sorted, hg19_idx, bw_path]
        subprocess.call(cmd)

        cmd_rm = ['rm', '-f', bedgraph_path]
        subprocess.call(cmd_rm)

        cmd_rm = ['rm', '-f', bedgraph_path_sorted]
        subprocess.call(cmd_rm)


def make_length_dict(path):
  length_dict = {}
  for line in open(path):
    a = line.split()
    length_dict[a[0]] = int(a[2])
  return length_dict


def make_chr_id(path):
  id_dict = {}
  for line in open(path):
    a = line.split()
    id_dict[a[4]] = a[0]
  return id_dict


# sort-bed H3k4me1.bedgraph > H3k4me1_sorted.bedgraph
# bedGraphToBigWig H3k79me2.bedgraph ../../hg19/hg19.fa.fai H3k79me2.bw
histone_list = ['H3K122ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K9ac', 'H3K9me3', 'H4K20me1']




if __name__ == '__main__':
    main()
