from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json

import numpy as np
import six
import tensorflow as tf

import tokenization
from modeling import BertConfig, create_initializer
from tokenization import convert_to_unicode, FullTokenizer


class OscarConfig(BertConfig):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 norm_ord=2,
                 encoder_layer=-2):
        """Constructs OscarConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
          entity_embedding_size: the size of pre-trained entity embeddings
        """
        super(OscarConfig, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range)
        self.encoder_layer = encoder_layer
        self.norm_ord = norm_ord

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = OscarConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NumberBatchEmbeddings(object):

    def __init__(self, vocab, embeddings, inv_vocab=None, mapping=None):
        self.vocab = vocab
        self.embeddings = embeddings
        self.inv_vocab = inv_vocab or {v: k for k, v in self.vocab.items()}
        self.mapping = mapping
        self.entity_embedding_size = embeddings.shape[1]
        self.entity_size = embeddings.shape[0]

    @classmethod
    def from_file(cls, entity_embedding_path):
        import gzip
        with gzip.open(entity_embedding_path, "rt") as reader:
            num_entities, size = next(reader).split(maxsplit=2)
            embeddings = np.zeros(shape=(int(num_entities) + 1, int(size)), dtype=np.float32)

            index = 1
            vocab = collections.OrderedDict()
            for line in reader:
                fields = line.split()
                entity = fields[0].replace('_', ' ')
                entity = entity.strip()
                vocab[entity] = index
                embeddings[index] = np.asarray(fields[1:], dtype=np.float32)
                index += 1
        return cls(vocab, embeddings)

    def create_subword_mapping(self, tokenizer: FullTokenizer):
        if not self.mapping:
            def tokenize(term):
                tokens = tokenizer.tokenize(term)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                return ids

            self.mapping = [tokenize(k) for k in self.vocab.keys()]
        return self.mapping


def fixed_compositional_distance(oscar_config, subword_embeddings, entity_embeddings, entity_mapping):
    entity_vocab_size, entity_embedding_size = entity_embeddings.shape

    with tf.variable_scope('reg/composition'):
        # entity_subword_embeddings = tf.nn.embedding_lookup(subword_embeddings, entity_mapping)

        weights = tf.get_variable(
            "composition_weights",
            shape=[entity_embedding_size, oscar_config.hidden_size],
            initializer=create_initializer(oscar_config.initializer_range))

        bias = tf.get_variable(
            "composition_bias", shape=[entity_embedding_size],
            initializer=tf.zeros_initializer())

        entity_subword_embeddings = []
        for idx, entity in enumerate(entity_mapping):
            entity_subword_sum = tf.add_n([subword_embeddings[subword] for subword in entity])
            comp = tf.matmul(tf.expand_dims(entity_subword_sum, axis=0), weights, transpose_b=True)
            comp = tf.nn.bias_add(comp, bias)
            comp = tf.norm(comp - entity_embeddings[idx], ord=2)
            entity_subword_embeddings.append(comp)

        loss = tf.add_n(entity_subword_embeddings)

        # entity_subword_embeddings = []
        # for entity in entity_mapping:
        #     entity_subword_sum = tf.add_n([subword_embeddings[subword] for subword in entity])
        #     entity_subword_embeddings.append(entity_subword_sum)
        #
        #
        # # entity_subword_embeddings = [tf.add_n(subword_embeddings[sws]) for sws in entity_mapping]
        # composed_entity_embeddings = tf.stack(entity_subword_embeddings)
        # composed_entity_embeddings = composed_entity_embeddings * weights + bias
        #
        # difference = tf.norm(composed_entity_embeddings - entity_embeddings, ord=oscar_config.norm_ord, axis=-1)
        # loss = tf.reduce_sum(difference)
    return loss

        #ex
        #
        #
        #
        # weights = tf.get_variable(
        #     "composition_weights",
        #     shape=[oscar_config.embedding_size, oscar_config.entity_embbedding_size],
        #     initializer=create_initializer(oscar_config.initializer_range))
        # bias = tf.get_variable(
        #     "composition_bias", shape=[oscar_config.entity_embbedding_size],
        #     initializer=tf.zeros_initializer())
        #
        # for subword_ids in entity_mapping:
        #     subword_sum = tf.reduce_sum(subword_embeddings, axis=-1)
        #     tf.matmul(subword_sum, weights)
        #     composed_entity_embeddings.append(weights * tf.reduce_sum(subword_embeddings[subword_ids], axis=-1) + bias)
        # composed_entity_embeddings = tf.stack(composed_entity_embeddings)
        #
        # norms = tf.norm(composed_entity_embeddings - entity_embeddings, ord='2', axis=-1)
        # return tf.reduce_mean(norms)
        #


class TrieNode(object):
    def __init__(self):
        self.values = set()
        self.children = collections.OrderedDict()

    def add(self, sequence, value):
        # print('Adding', sequence, 'with value', value)
        if not sequence:
            self.values.add(value)
        else:
            head = sequence[0]
            tail = sequence[1:]
            if head not in self.children:
                 self.children[head] = TrieNode()
            self.children[head].add(tail, value)


class SubwordTrie:
    def __init__(self, tokenizer, trie=None):
        self.tokenizer: tokenization.FullTokenizer = tokenizer
        self.trie = trie or TrieNode()

    @classmethod
    def from_vocab_file(cls, vocab_file, do_lower_case=True):
        return cls(tokenization.FullTokenizer(vocab_file, do_lower_case))

    def put_string(self, string, value):
        tokens = self.tokenizer.tokenize(string)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        self.trie.add(ids, value)
        return ids


def load_numberbatch(entity_embedding_path):
    import gzip
    with gzip.open(entity_embedding_path, "rt") as reader:
        num_entities, size = next(reader).split(maxsplit=2)
        embeddings = np.zeros(shape=(int(num_entities), int(size)), dtype=np.float32)
        index = 0
        vocab = collections.OrderedDict()
        for line in reader:
            fields = line.split()
            entity = fields[0]
            if all(ord(c) < 128 for c in entity):
                entity = entity.replace('_', ' ').strip()
                vocab[entity] = index
                embeddings[index] = np.asarray(fields[1:], dtype=np.float32)
                index += 1
    return vocab, embeddings


def find_entities(inputs, entity_trie):
    L = len(inputs)
    matches = []
    for i in range(L):
        node = entity_trie
        for j in range(i, L):
            node = node.children.get(inputs[j])
            if not node:
                break
            for value in node.values:
                matches.append((i, j + 1, value))
    return matches