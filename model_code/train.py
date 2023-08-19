import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import random
import tensorflow as tf
from tqdm import tqdm
from IPython.display import clear_output
import numpy as np
import pandas as pd
import time
import glob
import json
import functools
from model_enfo import Enformer
import datetime
import tensorboard
import random

'''
load_data
'''


def get_metadata(organism):
    # Keys:num_targets,num_atac,atac_length,seq_length, pool_width,crop_bp,target_length
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def tfrecord_files(organism, subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac-seq': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    atac = tf.io.decode_raw(example['atac-seq'], tf.float16)
    atac = tf.reshape(atac, (metadata['atac_length'], metadata['num_atac']))
    atac = tf.cast(atac, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target, (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return{
        'sequence': sequence,
        'atac-seq': atac,
        'target': target
    }


def organism_path(organism):
    return os.path.join('/local/ww/enformer/enformer_data/k562', organism)

def get_dataset(organism, subset, num_threads=8):
    metadata = get_metadata(organism)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads)
    # print(tfrecord_files(organism, subset))
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                          num_parallel_calls=num_threads)
    return dataset

'''
evaluation
'''

def _reduced_shape(shape, axis):
    if axis is None:
        return tf.TensorShape([])
    return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
    """Contains shared code for PearsonR and R2."""

    def __init__(self, reduce_axis=None, name='pearsonr'):
        """Pearson correlation coefficient.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation (say
            (0, 1). If not specified, it will compute the correlation across the
            whole tensor.
          name: Metric name.
        """
        super(CorrelationStats, self).__init__(name=name)
        self._reduce_axis = reduce_axis
        self._shape = None  # Specified in _initialize.

    def _initialize(self, input_shape):
        # Remaining dimensions after reducing over self._reduce_axis.
        self._shape = _reduced_shape(input_shape, self._reduce_axis)

        weight_kwargs = dict(shape=self._shape, initializer='zeros')
        self._count = self.add_weight(name='count', **weight_kwargs)
        self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
        self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
        self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                                 **weight_kwargs)
        self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
        self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                                 **weight_kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state.

        Args:
          y_true: Multi-dimensional float tensor [batch, ...] containing the ground
            truth values.
          y_pred: float tensor with the same shape as y_true containing predicted
            values.
          sample_weight: 1D tensor aligned with y_true batch dimension specifying
            the weight of individual observations.
        """
        if self._shape is None:
            # Explicit initialization check.
            self._initialize(y_true.shape)
        y_true.shape.assert_is_compatible_with(y_pred.shape)
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        self._product_sum.assign_add(
            tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

        self._true_sum.assign_add(
            tf.reduce_sum(y_true, axis=self._reduce_axis))

        self._true_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

        self._pred_sum.assign_add(
            tf.reduce_sum(y_pred, axis=self._reduce_axis))

        self._pred_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

        self._count.assign_add(
            tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

    def result(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    def reset_states(self):
        if self._shape is not None:
            tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                              for v in self.variables])


class PearsonR(CorrelationStats):
    """Pearson correlation coefficient.

    Computed as:
    ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
    """

    def __init__(self, reduce_axis=(0,), name='pearsonr'):
        """Pearson correlation coefficient.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation.
          name: Metric name.
        """
        super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                       name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        pred_mean = self._pred_sum / self._count

        covariance = (self._product_sum
                      - true_mean * self._pred_sum
                      - pred_mean * self._true_sum
                      + self._count * true_mean * pred_mean)

        true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
        pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
        tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
        correlation = covariance / tp_var

        return correlation


class R2(CorrelationStats):
    """R-squared  (fraction of explained variance)."""

    def __init__(self, reduce_axis=None, name='R2'):
        """R-squared metric.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation.
          name: Metric name.
        """
        super(R2, self).__init__(reduce_axis=reduce_axis,
                                 name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        total = self._true_squared_sum - self._count * tf.math.square(true_mean)
        residuals = (self._pred_squared_sum - 2 * self._product_sum
                     + self._true_squared_sum)

        return tf.ones_like(residuals) - residuals / total


class MetricDict:
    def __init__(self, metrics):
        self._metrics = metrics

    def update_state(self, y_true, y_pred):
        for k, metric in self._metrics.items():
            metric.update_state(y_true, y_pred)

    def result(self):
        return {k: metric.result() for k, metric in self._metrics.items()}

'''
def evaluate_model(model, dataset, head, max_steps=None):
    metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})

    @tf.function
    def predict(x):
        return model(x, is_training=False)[head]

    for i, batch in tqdm(enumerate(dataset)):
        if max_steps is not None and i > max_steps:
            break
        metric.update_state(batch['target'], predict(batch['sequence']))

    return metric.result()
'''

def evaluate_model(model, dataset, head, max_steps=None):
    metric = MetricDict({'PearsonR': PearsonR(reduce_axis = (0, 1))})

    @tf.function
    def predict(x, atac):
        return model(x, atac, is_training = False)[head]
    loss = 0
    len_dataset = 0
    for i, batch in tqdm(enumerate(dataset)):
        if max_steps is not None and i > max_steps:
            break
        predicted = predict(batch['sequence'], batch['atac-seq'])
        loss += tf.reduce_mean(
            tf.keras.losses.MSE(batch['target'], predicted))
        metric.update_state(batch['target'], predicted)
        len_dataset += 1
    loss = loss / len_dataset
    return metric.result(), loss
'''
train
'''


def create_step_function(model, optimizer):
  @tf.function
  def train_step(batch, head, optimizer_clip_norm_global=0.2):
    with tf.GradientTape() as tape:
      outputs = model(batch['sequence'], batch['atac-seq'], is_training=True)[head]
      # print(outputs)
      # print(outputs.shape)
      # print(batch['target'])
      loss = tf.reduce_mean(
        # tf.keras.losses.poisson(batch['target'], outputs))
        tf.keras.losses.MSE(batch['target'], outputs))
        # tf.keras.losses.huber(batch['target'], outputs))
    
    gradients = tape.gradient(loss, model.trainable_variables)

    # grads, variables = zip(gradients)
    gradients, global_norm = tf.clip_by_global_norm(gradients, 5)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
  return train_step


def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = r'/home/dut_ww/enformer/model_code/logs/' + current_time + '1536_4'

    max_correlation = 0

    summary_writer = tf.summary.create_file_writer(log_dir)

    human_dataset = get_dataset('atac', 'train').batch(1).repeat().prefetch(2)

    learning_rate = tf.Variable(0.0001, trainable=False, name='learning_rate')
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    num_warmup_steps = 2500
    target_learning_rate = 0.0001

    model = Enformer(channels=1536 // 2,  # // 4 Use 4x fewer channels to train faster.
                              num_heads = 8,
                              num_transformer_layers = 11,
                              pooling_type='max')
    train_step = create_step_function(model, optimizer)
    steps_per_epoch = 87868
    num_epochs = 300
    data_it = iter(human_dataset) 
    global_step = 0 
    for epoch_i in range(num_epochs):
        step = 0
        random_list = random.sample(list(range(steps_per_epoch)), 3000) 
        loss_train = 0
        for i in tqdm(range(steps_per_epoch)):  #tqdm()-进度条
            if i in random_list:
                global_step += 1
                if global_step > 1:
                    learning_rate_frac = tf.math.minimum(
                        1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))
                    learning_rate.assign(target_learning_rate * learning_rate_frac)

                batch_human = next(data_it)

                loss_human = train_step(batch=batch_human, head='human')
                loss_train += loss_human.numpy()
                with summary_writer.as_default():
                    tf.summary.scalar("loss_human step" + str(epoch_i), loss_human, step=step)
                step += 1
            else:
                batch_human = next(data_it)

        metrics_human, loss_human_val = evaluate_model(model,
                                       dataset=get_dataset('atac', 'valid').batch(1).prefetch(2),
                                       head='human',
                                       max_steps=1000)

        with summary_writer.as_default():
            tf.summary.scalar("huamn_PearsonR", metrics_human['PearsonR'].numpy().mean(), step=epoch_i)
            tf.summary.scalar("train_loss", loss_train/3000, step=epoch_i)
            tf.summary.scalar("val_loss", loss_human_val.numpy(), step=epoch_i)

            tf.summary.scalar("H3K122ac", metrics_human['PearsonR'].numpy()[0], step=epoch_i)
            tf.summary.scalar("H3k4me1", metrics_human['PearsonR'].numpy()[1], step=epoch_i)
            tf.summary.scalar("H3k4me2", metrics_human['PearsonR'].numpy()[2], step=epoch_i)
            tf.summary.scalar("H3k4me3", metrics_human['PearsonR'].numpy()[3], step=epoch_i)
            tf.summary.scalar("H3k27ac", metrics_human['PearsonR'].numpy()[4], step=epoch_i)
            tf.summary.scalar("H3k27me3", metrics_human['PearsonR'].numpy()[5], step=epoch_i)
            tf.summary.scalar("H3k36me3", metrics_human['PearsonR'].numpy()[6], step=epoch_i)
            tf.summary.scalar("H3k9ac", metrics_human['PearsonR'].numpy()[7], step=epoch_i)
            tf.summary.scalar("H3k9me3", metrics_human['PearsonR'].numpy()[8], step=epoch_i)
            tf.summary.scalar("H4k20me1", metrics_human['PearsonR'].numpy()[9], step=epoch_i)

        # End of epoch.
        print('loss_human', loss_human.numpy(),
              'learning_rate', optimizer.learning_rate.numpy()
              )
        if (epoch_i // 5) == 0 and epoch_i > 0:
            target_learning_rate = target_learning_rate / 1.4
        # learning_rate.assign(target_learning_rate)
        model.save_weights('model.ckpt')
        if max_correlation <= metrics_human['PearsonR'].numpy().mean():
            model.save_weights('model.ckpt')
            max_correlation = metrics_human['PearsonR'].numpy().mean()

if __name__ == '__main__':
    main()
