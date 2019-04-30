#! /bin/env python

import numpy as np
import sys


def parse_predicted_answers(logits_file):
    logits = np.loadtxt(logits_file)
    return np.argmax(logits, axis=-1)


def main(input_file):
    predictions = parse_predicted_answers(input_file)

    print("pair_id", "label", sep=",")
    for i, prediction in enumerate(predictions):
        print(i+1, prediction, sep=",")


if __name__ == '__main__':
    main(*sys.argv[1:])
