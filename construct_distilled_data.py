#!/usr/bin/python
#-*-coding:utf-8 -*-
#Version  : 1.0
#Filename : construct_distilled_data.py
from __future__ import print_function

import csv
import sys

def read_tsv(input_file, quotechar=None):
  """Reads a tab separated value file."""
  with open(input_file, "r") as f:
      #reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      reader = csv.reader((line.replace('\0','') for line in f), delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
          lines.append(line)
      return lines


def main(orig_file, pred_file, out_file):
    sents = read_tsv(orig_file)

    preds = read_tsv(pred_file)

    with open(out_file, "w") as f:
        for no, pred in preds:
            f.write("{}\t{}\n".format(sents[int(no)][0], pred))


if __name__ == "__main__":
    main(*sys.argv[1:])
