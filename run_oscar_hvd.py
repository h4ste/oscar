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
import optimization_hvd
import oscar

from run_oscar import input_fn_builder, model_fn_builder

import tensorflow as tf

import horovod.tensorflow as hvd

flags = tf.flags

FLAGS = flags.FLAGS


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
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        optimizer_fn=optimization_hvd.create_optimizer)

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
            entities=entities,
            embeddings=embeddings
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
