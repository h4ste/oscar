#! /bin/env python

import numpy as np
import sys
import json


def parse_predicted_answers(logits_file):
    logits = np.loadtxt(logits_file)
    return np.argmax(logits, axis=-1)


def parse_test_file(jsonl_file):
    ids = []
    with open(jsonl_file, 'r') as jsonl:
        for line in jsonl:
            data = json.loads(line)
            ids.append(data['pairID'])
    return ids


def main(test_file, logits_file):
    ids = parse_test_file(test_file)
    predictions = parse_predicted_answers(logits_file)

    print("pair_id", "label", sep=",")
    for id_, prediction in zip(ids, predictions):
        print(id_, prediction, sep=",")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
