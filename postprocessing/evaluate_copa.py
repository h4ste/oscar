import itertools
import xml.etree.cElementTree as ET
import sys

import numpy as np

examples = []


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def parse_predicted_answers(logits_file):
    with open(logits_file, 'r') as logits:
        lines = [float(line.split()[1]) for line in logits.readlines()]
    paired_logits = grouper(lines, 2)

    answers = []
    for alt_1, alt_2 in paired_logits:
        answers.append(1 if alt_1 > alt_2 else 2)

    return np.asarray(answers)


def parse_gold_answers(gold_file):
    tree = ET.parse(gold_file)
    root = tree.getroot()  # type: ET.Element

    answers = [int(item.attrib['most-plausible-alternative']) for item in root]

    return np.asarray(answers)


def evaluate(predictions, labels):
    if predictions.shape != labels.shape:
        print('Predictions had shape', predictions.shape, 'while labels had shape', labels.shape)
        raise AssertionError

    batch_n = predictions.shape[0]

    batch_tp = np.count_nonzero(predictions * labels)
    batch_tn = np.count_nonzero((predictions - 1) * (labels - 1))
    batch_fp = np.count_nonzero(predictions * (labels - 1))
    batch_fn = np.count_nonzero((predictions - 1))

    # Batch-level binary classification metrics
    batch_accuracy = (batch_tp + batch_tn) / batch_n
    batch_precision = batch_tp / (batch_tp + batch_fp)
    batch_recall = batch_tp / (batch_tp + batch_fn)
    batch_specificity = batch_tn / (batch_tn + batch_fp)
    batch_f1 = 2 * batch_precision * batch_recall / (batch_precision + batch_recall)

    print("Accuracy =", batch_accuracy)
    print("F1 =", batch_f1)
    print("Precision =", batch_precision)
    print("Recall =", batch_recall)
    print("Specificity =", batch_specificity)

    return batch_accuracy
    # print("TP =", batch_tp)
    # print("FP =", batch_fp)
    # print("FN =", batch_fn)
    # print("TN =", batch_tn)


def main(gold_file, logits_file):
    predictions = parse_predicted_answers(logits_file) - 1
    labels = parse_gold_answers(gold_file) - 1
    
    print("Evaluation Results:")
    acc = evaluate(predictions, labels)

    if acc < .5:
        print("\nInverted Evaluation Results")
        evaluate(1 - predictions, labels)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
