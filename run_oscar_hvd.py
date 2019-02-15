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
import modeling
import optimization_hvd as optimization
import oscar
import tokenization

import numpy as np

import tensorflow as tf

import horovod.tensorflow as hvd

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


def model_fn_builder(oscar_config, entity_embeddings, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
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

        entity_ids = features["entity_ids"]
        entity_positions = features["entity_positions"]
        entity_lengths = features["entity_lengths"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=oscar_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

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
                                                            entity_ids,
                                                            entity_positions,
                                                            entity_lengths)

        total_loss = masked_lm_loss + next_sentence_loss + entity_loss

        tvars = tf.trainable_variables()

        tf.logging.info("Placeholder V2:, %s",
                        [x for x in tf.get_default_graph().get_operations() if x.type == "PlaceholderV2"])
        tf.logging.info("Placeholder V1:, %s",
                        [x for x in tf.get_default_graph().get_operations() if x.type == "Placeholder"])

        gvars = tf.global_variables()
        tf.logging.info("*** Global Variables ***")
        for var in gvars:
            if var not in tvars:
                tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
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
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

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


def get_oscar_loss(oscar_config,  #type: oscar.OscarConfig
                   token_embeddings, entity_embeddings, entity_ids, entity_offsets, entity_lengths):
    """

    :type oscar_config: oscar.OscarConfig
    """
    with tf.variable_scope('oscar'):
        entity_embedding_table = tf.constant(entity_embeddings, name="entity_embedding_table", dtype=tf.float32)
        embedded_entities = tf.nn.embedding_lookup(entity_embedding_table, entity_ids, name='entity_embeddings')

        entity_embedding_shape = modeling.get_shape_list(embedded_entities)
        batch_size = entity_embedding_shape[0]
        max_entities = entity_embedding_shape[1]
        entity_width = entity_embedding_shape[2]

        composed_entities = slice_indexes(token_embeddings, entity_offsets, entity_lengths)
        tf.logging.info('Entity Slices: %s', composed_entities)
        # Composed entities are  [ (batch * max_entities) x token_embedding_size ]
        with tf.variable_scope('transform'):
            composed_entities = tf.layers.dense(composed_entities, units=entity_width,
                                              kernel_initializer=modeling.create_initializer(
                                                  oscar_config.initializer_range))
            composed_entities = modeling.layer_norm(composed_entities)

        composed_entities = tf.reshape(composed_entities, [batch_size, max_entities, entity_width])
        difference = tf.norm(embedded_entities - composed_entities, ord=oscar_config.norm_ord)
        mask = tf.to_float(tf.minimum(entity_lengths, 1))
        example_loss = tf.reduce_sum(difference * mask, axis=-1)
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


def slice_indexes(sequence_tensor, positions, lengths):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    positions_shape = modeling.get_shape_list(positions, expected_rank=2)
    num_positions = positions_shape[1]

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

    def slice(tensors):
        slice = tf.slice(flat_sequence_tensor, tensors[0], tensors[1])
        slice.set_shape([None, width])
        tf.logging.info('Slice::slice: %s', slice)
        aggregate_slice = tf.reduce_sum(slice, axis=0)
        tf.logging.info('Slice::aggregate_slice: %s', aggregate_slice)
        return aggregate_slice

    output_tensors = tf.map_fn(slice, [flat_2d_positions, flat_2d_lengths], dtype=tf.float32, infer_shape=True)
    tf.logging.info('Slice::output_tensors: %s', output_tensors)

    return output_tensors


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
                     is_training,
                     num_cpu_threads=4):
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

        def add_entities_and_decode(record):
            example = _decode_record(record, name_to_features)

            def find_entities(input_ids):
                entity_ids = np.zeros(shape=max_entities_per_seq, dtype=np.int32)
                entity_offsets = np.zeros(shape=max_entities_per_seq, dtype=np.int32)
                entity_lengths = np.zeros(shape=max_entities_per_seq, dtype=np.int32)

                _entities = oscar.find_entities(input_ids, entity_trie.trie)
                if _entities:
                    starts, ends, _ids = zip(*_entities)

                    tf.logging.log_every_n(tf.logging.INFO, 'Sequence: %s', input_ids.shape[0] * 1000,
                                           [t for t in entity_trie.tokenizer.convert_ids_to_tokens(input_ids) if t != '[PAD]'])
                    tf.logging.log_every_n(tf.logging.INFO, 'Entities: %s', input_ids.shape[0] * 1000,
                                           '; '.join(['%s@%d-%d' % (inv_entities[entity[2]], entity[0], entity[1]) for entity in _entities]))

                    assert all(start >= 0 for start in starts)
                    max_len = len(input_ids)
                    assert all(end <= max_len for end in ends)

                    entity_ids = np.zeros(shape=max_entities_per_seq, dtype=np.int32)
                    entity_offsets = np.zeros(shape=max_entities_per_seq, dtype=np.int32)
                    entity_lengths = np.zeros(shape=max_entities_per_seq, dtype=np.int32)

                    num_entities = np.minimum(len(_ids), max_entities_per_seq)
                    entity_ids[:num_entities] = _ids[:num_entities]
                    entity_offsets[:num_entities] = starts[:num_entities]
                    entity_lengths[:num_entities] = ends[:num_entities]

                return entity_ids, entity_offsets, entity_lengths - entity_offsets

            ids, offsets, lengths = tf.py_func(find_entities, [example['input_ids']], [tf.int32] * 3)

            ids.set_shape([max_entities_per_seq])
            offsets.set_shape([max_entities_per_seq])
            lengths.set_shape([max_entities_per_seq])

            example['entity_ids'] = ids
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


class EntityReporter(tf.train.SessionRunHook):

    def __init__(self, vocab_file, entity_embedding_path):
        self.entities, self.embeddings = oscar.load_numberbatch(entity_embedding_path)
        self.inv_entities = {v: k for k, v in self.entities.items()}
        self.entity_trie = oscar.SubwordTrie.from_vocab_file(vocab_file, True)
        for i, entity in enumerate(self.entities):
            path = self.entity_trie.put_string(entity, i)
            tf.logging.log_every_n(tf.logging.INFO, 'Adding %s as %s:%d', len(self.entities) / 100,
                                   entity, path, i)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(['bert/embeddings/word_embeddings', 'IteratorGetNext:0'])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        word_embeddings, input_ids = run_values.results

        for example_ids in input_ids:
            entities = [(t[0], t[1], [self.inv_entities[e] for e in t[2]]) for t in
                        oscar.find_entities(example_ids, self.entity_trie.trie)]
            tokens = self.entity_trie.tokenizer.convert_ids_to_tokens(example_ids)
            tf.logging.log_every_n(tf.logging.INFO, 'Sentence: %s',
                                   len(input_ids),
                                   ' '.join([tokenization.printable_text(x) for x in tokens if x != '[PAD]'][1:]))
            tf.logging.log_every_n(tf.logging.INFO, 'Entities (%d): {%s}',
                                   len(input_ids),
                                   len(entities),
                                   '; '.join(['%s@%d-%d' % (t[2], t[0], t[1]) for t in entities]))


def main(_):
    hvd.init()
    FLAGS.output_dir = FLAGS.output_dir if hvd.rank() == 0 else os.path.join(FLAGS.output_dir, str(hvd.rank()))
    FLAGS.num_train_steps = FLAGS.num_train_steps // hvd.size()
    FLAGS.num_warmup_steps = FLAGS.num_warmup_steps // hvd.size()

    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

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

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host),
        log_step_count_steps=25,
        session_config=config)

    model_fn = model_fn_builder(
        oscar_config=oscar_config,
        entity_embeddings=embeddings,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

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
            max_entities_per_seq=FLAGS.max_entities_per_seq,
            entity_trie=entity_trie,
            entities=entities
        )
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps, hooks=hooks)
        # hooks=[hook])

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False,
            max_entities_per_seq=FLAGS.max_entities_per_seq,
            entity_trie=entity_trie,
            entities=entities
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
