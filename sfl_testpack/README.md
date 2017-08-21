Shallow syntactic function labeller testpack
==============================================

This is a testpack for shallow syntactic function labeller, GSoC 2017 Apertium project.

More information can be found on Apertium Wiki: http://wiki.apertium.org/wiki/Shallow_syntactic_function_labeller

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

Script _setup.py_ adds all the needed files in language pair directory and changes all files with modes.

###### Arguments:

* _work_mode_: **-lb** for installing the labeller and changing modes, **-cg** for backwarding changes and using the original syntax module (sme-nob.syn.rlx.bin or kmr-eng.prob) in the pipeline.
* _lang_: **-sme** for installing/uninstalling the labeller only for sme-nob, **-kmr** - only for kmr-eng, **-all** - for both.

For example, this script will install the labeller and add it to the pipeline for both pairs:

```cmd
$ python setup.py -lb -all
```

And this script will backward modes changes for sme-nob:

```cmd
$ python setup.py -cg -sme
```
