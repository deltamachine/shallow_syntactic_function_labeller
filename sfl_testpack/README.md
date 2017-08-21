Shallow syntactic function labeller
=====================

This is Google Summer of Code 2017 project for Apertium.

Apertium Wiki: http://wiki.apertium.org/wiki/Shallow_syntactic_function_labeller

Repository for the whole project: https://github.com/deltamachine/shallow_syntactic_function_labeller

### Description

The shallow syntactic function labeller takes a string in Apertium stream format, parses it into a sequence of morphological tags and gives it to a classifier. The classifier is a simple RNN model trained on prepared datasets which were made from parsed syntax-labelled corpora (mostly UD-treebanks). The classifier analyzes the given sequence of morphological tags, gives a sequence of labels as an output and the labeller applies these labels to the original string.

In Apertium pipeline the labeller runs between morphological analyzer or disambiguator and pretransfer.

### Installation

#### Prerequisites
1. Python libraries:
	* DyNet (installation instructions can be found here: http://dynet.readthedocs.io/en/latest/python.html)
	* Streamparser (https://github.com/goavki/streamparser)
2. Precompiled language pairs which support the labeller (sme-nob, kmr-eng)

#### How to install a testpack
NB: currently this testpack contains syntax modules only for sme-nob and kmr-eng.

```cmd
$ git clone https://github.com/deltamachine/sfl_testpack.git
$ cd sfl_testpack
```

Script setup.py adds all the needed files in language pair directory and changes all files with modes.

#### Arguments: ###

* work_mode: -lb for installing the labeller and changing modes, -cg for backwarding changes and using the original syntax module (sme-nob.syn.rlx.bin or kmr-eng.prob) in the pipeline.
* lang: -sme for installing/uninstalling the labeller only for sme-nob, -kmr - only for kmr-eng, -all - for both.

For example, this script will install the labeller and add it to the pipeline for both pairs:

```cmd
$ python setup.py -lb -all
```

And this script will backward modes changes for sme-nob:

```cmd
$ python setup.py -cg -sme
```
