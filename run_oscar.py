# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function

import gpu_environment
import modeling
import optimization
import oscar

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "oscar_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "entity_embedding_file", None,
    "white space separated entity embeddings.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_entities_per_seq", 32,
    "The maximum number of entities to recognize in each input sequence. "
    "Sequences with a greater number of entities than this will be truncated, and sequences with fewer"
    "will be padded.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")

flags.DEFINE_bool("report_loss", False, "Whether to report total loss during training.")

flags.DEFINE_integer("report_loss_iters", 10, "How many iterations between loss reports")

flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")

flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")

flags.DEFINE_bool("allow_subsumed_entities", False,
                  "Whether to allow subsumed entities (e.g., whether to allow 'dog' to be recognized if 'hot dog' is "
                  "recognized.)")

flags.DEFINE_bool("allow_masked_entities", False, "Whether to allow entities to span across masked tokens.")

flags.DEFINE_enum("composition_method", "linear", ["linear", "RAN", "TRAN"],
                  "How to compose token embeddings for oscar loss.")

flags.DEFINE_enum("oscar_regularization", "mean", ["sum", "mean"],
                  "How to aggregate oscar regularization loss.")

flags.DEFINE_enum("oscar_distance", "l2", ["l1", "l2", "cosine"],
                  "How to measure distance between compositional and pre-trained entity embeddings.")

flags.DEFINE_float("oscar_smoothing", 1., "Factor applied to smooth Oscar regularization")


def angular_cosine_distance(composed, pretrained):
    composed_norm = batch_norm(composed, l2_norm)
    pretrained_norm = batch_norm(pretrained, l1_norm)
    cosine_similarity = tf.reduce_sum(tf.multiply(composed_norm, pretrained_norm), axis=-1)
    angular_distance = tf.acos(cosine_similarity) / math.pi
    return angular_distance


# Numerically stable norm calculation from https://github.com/tensorflow/tensorflow/issues/12071


@function.Defun(tf.float32, tf.float32)
def l1_norm_grad(x, dy):
    return dy*(x/(tf.norm(x, ord=1)+1.0e-19))


@function.Defun(tf.float32, tf.float32)
def l2_norm_grad(x, dy):
    return dy*(x/(tf.norm(x, ord=2)+1.0e-19))


@function.Defun(tf.float32, grad_func=l1_norm_grad)
def l1_norm(x):
    return tf.norm(x, ord=1)


@function.Defun(tf.float32, grad_func=l2_norm_grad)
def l2_norm(x):
    return tf.norm(x, ord=2)


def batch_norm(x, norm_fn):
    return tf.map_fn(norm_fn, x)


distance_metrics = {
    "cosine": angular_cosine_distance,
    "l1": lambda composed, pretrained, axis=None: batch_norm(composed - pretrained, l1_norm),
    "l2": lambda composed, pretrained, axis=None: batch_norm(composed - pretrained, l2_norm),
}


# report samples/sec, total loss and learning rate during training
# noinspection PyAttributeOutsideInit
class _LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, global_batch_size, display_every=10, hvd_rank=-1):
        self.global_batch_size = global_batch_size
        self.display_every = display_every
        self.hvd_rank = hvd_rank

    def after_create_session(self, session, coord):
        self.elapsed_secs = 0.
        self.count = 0
        print('  Step samples/sec   MLM Loss  NSP Loss  OSC Loss  Loss  Learning-rate')

    def before_run(self, run_context):
        self.t0 = time.time()
        return tf.train.SessionRunArgs(
            fetches=['step_update:0', 'total_loss:0',
                     'learning_rate:0', 'nsp_loss:0',
                     'mlm_loss:0', 'osc_loss:0'])

    def after_run(self, run_context, run_values):
        self.elapsed_secs += time.time() - self.t0
        self.count += 1
        global_step, total_loss, lr, nsp_loss, mlm_loss, osc_loss = run_values.results
        print_step = global_step + 1  # One-based index for printing.
        if print_step == 1 or print_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = self.global_batch_size / dt
            if self.hvd_rank >= 0:
                print('%2d :: %6i %11.1f %10.4g %10.4g %10.4g %10.4g     %6.4e' %
                      (self.hvd_rank, print_step, img_per_sec, mlm_loss, nsp_loss, osc_loss, total_loss, lr))
            else:
                print('%6i %11.1f %10.4g %10.4g %10.4g %10.4g     %6.4e' %
                      (print_step, img_per_sec, mlm_loss, nsp_loss, osc_loss, total_loss, lr))
            self.elapsed_secs = 0.
            self.count = 0


def model_fn_builder(oscar_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings,
                     hvd=None):
    """Returns `model_fn` closure for TPUEstimator."""

    # noinspection PyUnusedLocal
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s, tensor = %s, type = %s, feature = %s" % (
                name, features[name].shape, features[name].name, type(features[name]), features[name]))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        entity_embeddings = features["entity_embeddings"]
        entity_positions = features["entity_positions"]
        entity_lengths = features["entity_lengths"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=oscar_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            compute_type=tf.float16 if FLAGS.use_fp16 else tf.float32)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            oscar_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
            oscar_config, model.get_pooled_output(), next_sentence_labels)

        (entity_loss, entity_example_loss) = get_oscar_loss(oscar_config,
                                                            model.get_all_encoder_layers()[oscar_config.encoder_layer],
                                                            entity_embeddings,
                                                            entity_positions,
                                                            entity_lengths,
                                                            compute_type=tf.float16 if FLAGS.use_fp16 else tf.float32)

        masked_lm_loss = tf.identity(masked_lm_loss, "mlm_loss")
        next_sentence_loss = tf.identity(next_sentence_loss, "nsp_loss")
        entity_loss = tf.identity(FLAGS.oscar_smoothing * entity_loss, "osc_loss")

        total_loss = masked_lm_loss + next_sentence_loss + entity_loss
        total_loss = tf.identity(total_loss, name='total_loss')

        tvars = tf.trainable_variables()

        gvars = tf.global_variables()
        tf.logging.info("*** Global Variables ***")
        for var in gvars:
            if var not in tvars:
                tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("*** Trainable Variables ***")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  %d :: name = %s, shape = %s%s", 0 if hvd is None else hvd.rank(), var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                hvd, FLAGS.use_fp16)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            # noinspection PyShadowingNames
            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                          masked_lm_weights, next_sentence_example_loss,
                          next_sentence_log_probs, next_sentence_labels,
                          entity_example_loss):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                                 [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
                next_sentence_predictions = tf.argmax(
                    next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.metrics.accuracy(
                    labels=next_sentence_labels, predictions=next_sentence_predictions)
                next_sentence_mean_loss = tf.metrics.mean(
                    values=next_sentence_example_loss)

                example_mean_loss = tf.metrics.mean(
                    values=entity_example_loss)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                    "entity_example_loss": example_mean_loss
                }

            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels,
                entity_example_loss
            ])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % mode)

        return output_spec

    return model_fn


def _matvecmul(mat, vec, transpose_a=False, transpose_b=False):
    return tf.squeeze(tf.matmul(mat,
                                tf.expand_dims(vec, axis=-1),
                                transpose_a=transpose_a,
                                transpose_b=transpose_b), axis=-1)


def get_oscar_loss(oscar_config,
                   token_embeddings, embedded_entities, entity_offsets, entity_lengths,
                   compute_type=tf.float32):
    with tf.variable_scope('oscar', custom_getter=gpu_environment.get_custom_getter(compute_type)):
        # entity_embedding_table = tf.constant(entity_embeddings, name="entity_embedding_table", dtype=tf.float32)
        # embedded_entities = tf.nn.embedding_lookup(entity_embedding_table, entity_ids, name='entity_embeddings')
        entity_embedding_shape = modeling.get_shape_list(embedded_entities)
        batch_size = entity_embedding_shape[0]
        max_entities = entity_embedding_shape[1]
        entity_width = entity_embedding_shape[2]

        slice_fn = None
        with tf.variable_scope('compose'):
            if FLAGS.composition_method == 'linear':
                comp_weights = tf.get_variable(
                    "weights", [oscar_config.hidden_size, oscar_config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02),
                    dtype=compute_type)

                comp_bias = tf.get_variable(
                    "bias", oscar_config.hidden_size, initializer=tf.zeros_initializer(),
                    dtype=compute_type)

                def slice_fn(slice_):
                    sum_slice = tf.reduce_sum(slice_, axis=0)  # , keepdims=True)
                    tf.logging.info("linear::slice_fn::sum_slice: %s", sum_slice)
                    composed_entity = _matvecmul(comp_weights, sum_slice)
                    tf.logging.info("linear::slice_fn::scaled_slice: %s", composed_entity)
                    composed_entity = composed_entity + comp_bias
                    # composed_entity = tf.matmul(composed_entity, comp_weights)
                    # composed_entity = tf.nn.bias_add(composed_entity, comp_bias)
                    return composed_entity

            elif FLAGS.composition_method == 'RAN':
                content_weight = tf.get_variable(
                    "content_weights", [oscar_config.hidden_size, oscar_config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02),
                    dtype=compute_type)

                input_weights = tf.get_variable(
                    "input_weights", [2 * oscar_config.hidden_size, oscar_config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02),
                    dtype=compute_type)

                input_bias = tf.get_variable(
                    "input_bias", oscar_config.hidden_size,
                    initializer=tf.zeros_initializer(),
                    dtype=compute_type)

                forget_weights = tf.get_variable(
                    "forget_weights", [2 * oscar_config.hidden_size, oscar_config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02),
                    dtype=compute_type)

                forget_bias = tf.get_variable(
                    "forget_bias", oscar_config.hidden_size,
                    initializer=tf.zeros_initializer(),
                    dtype=compute_type)

                def ran_cell(prev, input_):
                    prev_state, prev_output = prev
                    content = _matvecmul(content_weight, input_)
                    flat_input = tf.concat([input_, prev_output], axis=-1)
                    input_gate = tf.nn.sigmoid(_matvecmul(input_weights, flat_input, transpose_a=True) + input_bias)
                    forget_gate = tf.nn.sigmoid(_matvecmul(forget_weights, flat_input, transpose_a=True) + forget_bias)
                    state = input_gate * content + forget_gate * prev_state
                    output = tf.nn.tanh(state)
                    return state, output

                initial_state = (tf.zeros(oscar_config.hidden_size, dtype=compute_type),
                                 tf.zeros(oscar_config.hidden_size, dtype=compute_type))

                def slice_fn(slice_):
                    state, output = tf.scan(fn=ran_cell, elems=slice_, initializer=initial_state)
                    tf.logging.info("RAN::slice_fn::state (ignored): %s", state)
                    tf.logging.info("RAN::slice_fn::output: %s", output)
                    return output[-1]

            elif FLAGS.composition_method == 'TRAN':
                input_weights = tf.get_variable(
                    "input_weights", [2 * oscar_config.hidden_size, oscar_config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02),
                    dtype=compute_type)

                input_bias = tf.get_variable(
                    "input_bias", oscar_config.hidden_size,
                    initializer=tf.zeros_initializer(),
                    dtype=compute_type)

                forget_weights = tf.get_variable(
                    "forget_weights", [2 * oscar_config.hidden_size, oscar_config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02),
                    dtype=compute_type)

                forget_bias = tf.get_variable(
                    "forget_bias", oscar_config.hidden_size,
                    initializer=tf.zeros_initializer(),
                    dtype=compute_type)

                def ran_cell(prev_output, input_):
                    flat_input = tf.concat([input_, prev_output], axis=-1)
                    input_gate = tf.nn.sigmoid(_matvecmul(input_weights, flat_input, transpose_a=True) + input_bias)
                    forget_gate = tf.nn.sigmoid(_matvecmul(forget_weights, flat_input, transpose_a=True) + forget_bias)
                    state = input_gate * input_ + forget_gate * prev_output
                    return state

                initial_state = tf.zeros(oscar_config.hidden_size, dtype=compute_type)

                def slice_fn(slice_):
                    state = tf.scan(ran_cell, slice_, initial_state)
                    return state[-1]

            else:
                raise NotImplementedError('Composition method %s is not supported.' % FLAGS.composition_method)

        composed_entities = slice_indexes(token_embeddings, entity_offsets, entity_lengths, slice_fn,
                                          dtype=compute_type)
        tf.logging.info('Entity Slices: %s', composed_entities)

        # Composed entities are  [ (batch * max_entities) x token_embedding_size ]
        with tf.variable_scope('project'):
            proj_weights = tf.get_variable(
                "weights", [oscar_config.hidden_size, entity_width],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                dtype=compute_type)

            proj_bias = tf.get_variable(
                "bias", [entity_width], initializer=tf.zeros_initializer(),
                dtype=compute_type)

            composed_entities = tf.nn.bias_add(tf.matmul(composed_entities, proj_weights), proj_bias)

        with tf.variable_scope('loss'):
            composed_entities = tf.reshape(composed_entities, [batch_size, max_entities, entity_width])
            composed_entities = tf.cast(composed_entities, tf.float32)
            difference = distance_metrics[FLAGS.oscar_distance](composed_entities, embedded_entities)
            mask = tf.to_float(tf.minimum(entity_lengths, 1))
            # If we have no entities, we set num_entities = 1 to avoid division by zero
            num_entities = tf.maximum(tf.reduce_sum(mask, axis=-1), 1)
            # 1/2 * average l2 difference
            if FLAGS.oscar_regularization.lower() == 'mean':
                example_loss = tf.reduce_sum(difference * mask, axis=-1) / (2 * num_entities)
            elif FLAGS.oscar_regularization.lower() == 'sum':
                example_loss = tf.reduce_sum(difference * mask, axis=-1) * .5
            else:
                raise NotImplementedError("Unsupported regularization method %s", FLAGS.oscar_regularization)

            loss = tf.reduce_mean(example_loss, axis=-1)

            return loss, example_loss


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)
    tf.logging.info('Masked LM Input Tensor: %s', input_tensor)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return loss, per_example_loss, log_probs


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, log_probs


def slice_indexes(sequence_tensor, positions, lengths, slice_fn, dtype=tf.float32):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    tf.logging.info('Slice::sequence_tensor: %s', sequence_tensor)
    tf.logging.info('Slice::positions: %s', positions)
    tf.logging.info('Slice::lengths: %s', lengths)

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    tf.logging.info('Slice::flat_offsets: %s', flat_offsets)
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    tf.logging.info('Slice::flat_positions: %s', flat_positions)
    flat_2d_positions = tf.stack([flat_positions, tf.broadcast_to(0, flat_positions.shape)], axis=-1)
    tf.logging.info('Slice::flat_2d_positions: %s', flat_2d_positions)
    flat_lengths = tf.reshape(lengths, [-1])
    tf.logging.info('Slice::flat_lengths: %s', flat_lengths)
    flat_2d_lengths = tf.stack([flat_lengths, tf.broadcast_to(width, flat_lengths.shape)], axis=-1)
    tf.logging.info('Slice::flat_2d_lengths: %s', flat_2d_lengths)
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])

    tf.logging.info('Slice::flat_sequence_tensor: %s', flat_sequence_tensor)

    empty_slice = tf.zeros(width, dtype=dtype)

    def slice_fn_wrapper(tensors):
        position, length = tensors

        def true_fn():
            slice_ = tf.slice(flat_sequence_tensor, position, length)
            slice_.set_shape([None, width])
            tf.logging.info('Slice Indices::slice_fn_wrapper::true::slice: %s', slice_)
            transformed_slice = slice_fn(slice_)
            tf.logging.info('Slice Indices::slice_fn_wrapper::true::transformed_slice: %s', transformed_slice)
            return transformed_slice

        def false_fn():
            tf.logging.info('Slice Indices::slice_fn_wrapper::false::empty_slice: %s', empty_slice)
            return empty_slice

        tf.logging.info('Slice Indices::slice_fn_wrapper::position: %s', position)
        tf.logging.info('Slice Indices::slice_fn_wrapper::length: %s', length)

        return tf.cond(tf.greater(length[0], 0), true_fn, false_fn)

    transformed_slices = tf.map_fn(slice_fn_wrapper, (flat_2d_positions, flat_2d_lengths), dtype=dtype,
                                   infer_shape=True)
    tf.logging.info('Slice Indices::transformed_slices: %s', transformed_slices)

    return transformed_slices


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    tf.logging.info('Gather::sequence_tensor: %s', sequence_tensor)
    tf.logging.info('Gather::positions: %s', positions)

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    tf.logging.info('Gather::flat_offsets: %s', flat_offsets)
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    tf.logging.info('Gather::flat_positions: %s', flat_positions)
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    tf.logging.info('Gather::flat_sequence_tensor: %s', flat_sequence_tensor)
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     max_entities_per_seq,
                     entity_trie,
                     entities,
                     embeddings,
                     is_training,
                     num_cpu_threads=4,
                     hvd=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    inv_entities = {v: k for k, v in entities.items()}

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.FixedLenFeature([1], tf.int64)
            # "entity_ids":
            #     tf.FixedLenFeature([max_entities_per_seq], tf.int64),
            # "entity_offsets":
            #     tf.FixedLenFeature([max_entities_per_seq], tf.int64),
            # "entity_lengths":
            #     tf.FixedLenFeature([max_entities_per_seq], tf.int64)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            if hvd is not None:
                d = d.shard(hvd.size(), hvd.rank())
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        embedding_width = embeddings.shape[1]

        def add_entities_and_decode(record):
            example = _decode_record(record, name_to_features)

            def find_entities(input_ids):
                entity_embeddings = np.zeros(shape=[max_entities_per_seq, embedding_width], dtype=np.float32)
                entity_offsets = np.zeros(shape=max_entities_per_seq, dtype=np.int32)
                entity_lengths = np.zeros(shape=max_entities_per_seq, dtype=np.int32)

                if FLAGS.allow_subsumed_entities:
                    _entities = oscar.find_entities(input_ids, entity_trie.trie)

                    tf.logging.log_first_n(tf.logging.INFO, 'Sequence: %s\nAll Entities: %s', 5,
                                           [t for t in entity_trie.tokenizer.convert_ids_to_tokens(input_ids) if
                                            t != '[PAD]'],
                                           '; '.join(
                                               ['%s@%d-%d' % (inv_entities[entity[2]], entity[0], entity[1]) for entity
                                                in _entities]))
                else:
                    _entities = oscar.find_longest_entities(input_ids, entity_trie.trie)

                    tf.logging.log_first_n(tf.logging.INFO, 'Sequence: %s\nLongest Entities: %s\nAll Entities: %s', 5,
                                           [t for t in entity_trie.tokenizer.convert_ids_to_tokens(input_ids) if
                                            t != '[PAD]'],
                                           '; '.join(
                                               ['%s@%d-%d' % (inv_entities[entity[2]], entity[0], entity[1]) for entity
                                                in _entities]),
                                           '; '.join(
                                               ['%s@%d-%d' % (inv_entities[entity[2]], entity[0], entity[1]) for entity
                                                in oscar.find_entities(input_ids, entity_trie.trie)])
                                           )

                if _entities:
                    starts, ends, _ids = zip(*_entities)

                    # assert all(start >= 0 for start in starts)
                    # max_len = len(input_ids)
                    # assert all(end <= max_len for end in ends)

                    num_entities = np.minimum(len(_ids), max_entities_per_seq)
                    entity_embeddings[:num_entities] = [embeddings[_id] for _id in _ids[:num_entities]]
                    entity_offsets[:num_entities] = starts[:num_entities]
                    entity_lengths[:num_entities] = ends[:num_entities]

                return entity_embeddings, entity_offsets, entity_lengths - entity_offsets

            if FLAGS.allow_masked_entities:
                def unmask_find_entities(input_ids, mask_positions, mask_ids):
                    unmasked_input_ids = np.copy(input_ids)
                    for position, id_ in (x for x in zip(mask_positions, mask_ids) if x[0] != 0):
                        unmasked_input_ids[position] = id_

                    tf.logging.log_first_n(tf.logging.INFO, 'Mask Positions: %s\nMask Ids: '
                                                            '%s\nMasked Tokens: %s\nMasked Sequence: %s\nUnmasked '
                                                            'Sequence: %s',
                                           5,
                                           mask_positions,
                                           mask_ids,
                                           [t for t in entity_trie.tokenizer.convert_ids_to_tokens(mask_ids)],
                                           [t for t in entity_trie.tokenizer.convert_ids_to_tokens(input_ids) if
                                            t != '[PAD]'],
                                           [t for t in entity_trie.tokenizer.convert_ids_to_tokens(unmasked_input_ids)
                                            if
                                            t != '[PAD]'])

                    return find_entities(unmasked_input_ids)

                _embeddings, offsets, lengths = tf.py_func(unmask_find_entities,
                                                           [example['input_ids'], example['masked_lm_positions'],
                                                            example['masked_lm_ids']],
                                                           [tf.float32, tf.int32, tf.int32])
            else:
                _embeddings, offsets, lengths = tf.py_func(find_entities, [example['input_ids']],
                                                           [tf.float32, tf.int32, tf.int32])

            _embeddings.set_shape([max_entities_per_seq, embedding_width])
            offsets.set_shape([max_entities_per_seq])
            lengths.set_shape([max_entities_per_seq])

            example['entity_embeddings'] = _embeddings
            example['entity_positions'] = offsets
            example['entity_lengths'] = lengths

            return example

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                add_entities_and_decode,  # lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))

        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if FLAGS.horovod:
        import horovod.tensorflow as hvd
        hvd.init()

    oscar_config = oscar.OscarConfig.from_json_file(FLAGS.oscar_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    entities, embeddings = oscar.load_numberbatch(FLAGS.entity_embedding_file)
    entity_trie = oscar.SubwordTrie.from_vocab_file(FLAGS.vocab_file, True)
    for i, entity in enumerate(entities):
        path = entity_trie.put_string(entity, i)
        tf.logging.log_every_n(tf.logging.INFO, 'Adding %s as %s:%d', len(entities) / 100,
                               entity, path, i)

    config = tf.ConfigProto()
    if FLAGS.horovod:
        config.gpu_options.visible_device_list = str(hvd.local_rank())
    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        config.gpu_options.allow_growth = True

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        session_config=config,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps if not FLAGS.horovod or hvd.rank() == 0 else None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host),
        # This variable controls how often estimator reports examples/sec.
        # Default value is every 100 steps.
        # When --report_loss is True, we set to very large value to prevent
        # default info reporting from estimator.
        # Ideally we should set it to None, but that does not work.
        log_step_count_steps=10000 if FLAGS.report_loss else 100)

    model_fn = model_fn_builder(
        oscar_config=oscar_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        hvd=None if not FLAGS.horovod else hvd)

    training_hooks = []
    if FLAGS.horovod and hvd.size() > 1:
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    if FLAGS.report_loss:
        global_batch_size = FLAGS.train_batch_size if not FLAGS.horovod else FLAGS.train_batch_size * hvd.size()
        training_hooks.append(
            _LogSessionRunHook(global_batch_size, FLAGS.report_loss_iters, -1 if not FLAGS.horovod else hvd.rank()))
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    # hook = EntityReporter(FLAGS.vocab_file, FLAGS.entity_embedding_file)
    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True,
            hvd=None if not FLAGS.horovod else hvd,
            max_entities_per_seq=FLAGS.max_entities_per_seq,
            entity_trie=entity_trie,
            entities=entities,
            embeddings=embeddings
        )
        estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=FLAGS.num_train_steps)
        # hooks=[hook])

    if FLAGS.do_eval and (not FLAGS.horovod or hvd.rank() == 0):
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False,
            hvd=None if not FLAGS.horovod else hvd,
            max_entities_per_seq=FLAGS.max_entities_per_seq,
            entity_trie=entity_trie,
            entities=entities,
            embeddings=embeddings
        )

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("oscar_config_file")
    flags.mark_flag_as_required("entity_embedding_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
