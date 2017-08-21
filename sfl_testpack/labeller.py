# -*- coding: utf-8 -*-

"""
The machine-learned syntax module for Apertium.
In the pipeline the labeller runs between morphological disambiguator and pretransfer.
More information about the project can be found here: http://wiki.apertium.org/wiki/Shallow_syntactic_function_labeller

=== Installation ===
Use setup.py and instructions given in README file.

=== Dependencies ===
* apertium-lang-embeddings: file with embeddings for all morphological tags the model can work with
* apertium-lang-tags: a list of morphological tags the model can work with
* apertium-lang-int2syntax: integer codes for the syntax labels
* apertium-lang-syntax: pretrained network which the labeller restores
"""


import re
import json
import dynet as dy
import numpy as np
from itertools import product
from streamparser import parse, readingToString


class SimpleRNNNetwork:
    def __init__(self, rnn_num_of_layers, vectors, state_size):
        """ Simple RNN network which is able to do syntactic labelling.

                Args:
                    * rnn_num_of_layers (int): number of network's layers
                    * vectors (list): list with all tags' embeddings
                    * state_size (int): network's state size
        """

        self.model = dy.ParameterCollection()
        self.embeddings = self.model.add_lookup_parameters(
            (len(vectors), len(vectors[0])))
        
        self.embeddings.init_from_array(np.array(vectors))
        
        self.RNN = dy.LSTMBuilder(
            rnn_num_of_layers, len(
                vectors[0]), state_size, self.model)
        
        self.output_w = self.model.add_parameters((len(vectors), state_size))
        self.output_b = self.model.add_parameters((len(vectors)))

    def _preprocess_input(self, string, vocab):
        """ Preprocess input sequence of morphological tags: split the string, add <EOS> tag, then
        transform every word into a sum of its tags' embeddings and return a list with words' vectors. """

        string = string.split() + ['<EOS>']
        input_list = []

        for i in range(len(string)):
            string[i] = re.sub('><', '>!<', string[i]).split('!')
            word = np.array([dy.lookup(self.embeddings, vocab[elem])
                             for elem in string[i]])
            word = np.sum(word, axis=0)

            input_list.append(word)

        return input_list

    def _run_rnn(self, init_state, input_vecs):
        """ Well, run RNN. """

        s = init_state
        states = s.add_inputs(input_vecs)
        rnn_outputs = [s.output() for s in states]

        return rnn_outputs

    def _get_probs(self, rnn_output):
        """ Map the computed output of the RNN to a probability distribution over candidate tags
        by applying a softmax transformation. """

        output_w = dy.parameter(self.output_w)
        output_b = dy.parameter(self.output_b)
        probs = dy.softmax(output_w * rnn_output + output_b)

        return probs

    def _predict(self, probs, int2syntax):
        """ Pick the maximum likelihood tag given the prob distribution. """

        probs = probs.value()
        predicted_tag = int2syntax[str(probs.index(max(probs)))]

        return predicted_tag

    def generate(self, input_string, vocab, int2syntax):
        """ Generate a candidate output given the input based on the current state of the network. """

        dy.renew_cg()

        input_string = self._preprocess_input(input_string, vocab)

        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, input_string)

        output_string = []

        for rnn_output in rnn_outputs:
            probs = self._get_probs(rnn_output)
            predicted_tag = self._predict(probs, int2syntax)
            output_string.append(predicted_tag)

        output_string = ' '.join(output_string)

        return output_string.replace('<EOS>', '')


class DoubleArray:
    def __init__(self, list1, list2):
        """ Implement a two-arrays structure. """

        self.list1 = list1
        self.list2 = list2

    def add(self, elem1, elem2):
        """ Add a first element in the first list and a second element in the second list. """

        self.list1.append(elem1)
        self.list2.append(elem2)

    def tags_to_reading(self, elem):
        """ Given an element from the second list, return an element with the same index from the first list. """

        i = self.list2.index(elem)
        return self.list1[i]

    def replace_reading(self, elem, new_elem):
        """ Given an element from the second list, replace an element with the same index from the first
        list with a new element. """

        i = self.list2.index(elem)
        self.list1[i] = new_elem


def prepare_data():
    """ Read a file with embeddings and a file with {integer_code: syntax_label} dictionary and
    create needed files for RNN network.

    Returns:
        * vectors (list): list with all tags' embeddings
        * vocab (dictionary): dictionary where keys are tags and values are embeddings
        * int2syntax (dictionary): dictionary where keys are integer codes and values are syntax labels
    """

    vectors = []
    vocab = {}

    with open('apertium-language-embeddings.vec', 'r', encoding='utf-8') as file:
        f = file.readlines()

    with open('apertium-language-int2syntax.json', 'r', encoding='utf-8') as jsonfile:
        s = jsonfile.read()

    for i, line in enumerate(f):
        word = line.split()
        vocab[word[0]] = i
        vectors.append(list(map(float, word[1:])))

    int2syntax = json.loads(s)

    return vectors, vocab, int2syntax


def split_sentences(string):
    """ Take a string in Apertium stream format and split it into sentences. Return a list with all sentences. """

    string = re.sub('!', '!<eos>', string)
    string = re.sub(re.escape('^?/?<sent>$'), '^?/?<sent>$' + '<eos>', string)
    string = re.sub(
        re.escape('^../..<sent>$'),
        '^../..<sent>$' +
        '<eos>',
        string)
    string = re.sub(re.escape('^./.<sent>$'), '^./.<sent>$' + '<eos>', string)
    sentences = string.split('<eos>')
    sentences = [elem.strip() for elem in sentences[:-1]]

    return sentences


def parse_asf(string):
    """ Take a string in Apertium stream format and parse it into a sequence (or into a few possible sequences)
    of morphological tags using streamparser library.

    Returns:
        * sequences (list): list with all possible readings' sequences. The number of sequences depends on the number
        of words with > 1 readings.
        * combinations (DoubleArray): DoubleArray where the first array contains lemmas + tags and the second - only tags.
    """

    units = parse(c for c in string)
    combinations = DoubleArray([], [])
    options = {}

    for unit in units:
        tags = []

        for reading in unit.readings:
            joined_lu_tags = []
            word = readingToString(reading)

            for i in range(len(reading)):
                t = '<' + '><'.join(reading[i][1]) + '>'
                combinations.add(reading[i][0] + t, t)
                joined_lu_tags.append(t)

            tags.append(' '.join(joined_lu_tags))

        options[str(unit)] = tags

    elements = [options[re.sub('.*?\^', '', key)]
                for key in string.split('$')[:-1]]
    elements = product(*elements)
    sequences = [' '.join(elem) for elem in elements]

    return sequences, combinations


def replace_unknown_tags(checked_sequences):
    """ Delete all tags in sequences which model doesn't know """

    with open('apertium-language-tags.txt', 'r', encoding='utf-8') as file:
        morph_tags = file.read().split()

    for i in range(len(checked_sequences)):
        sequence_tags = checked_sequences[i].split()

        for j in range(len(sequence_tags)):
            sequence_tags[j] = re.sub('><', '>!<', sequence_tags[j]).split('!')

            for elem in sequence_tags[j]:
                if elem not in morph_tags:
                    checked_sequences[i] = re.sub(
                        elem, '', checked_sequences[i])

    return checked_sequences


def add_functions(rnn, string, vocab, int2syntax, sequences, combinations):
    """ Add syntactic function labels to a sentence from the original string.

    Args:
        * rnn (SimpleRNNNetwork): pretrained RNN Network
        * string (string): sequence of morpholodical tags (one sentence)
        * vocab (dictionary): dictionary where keys are tags and values are embeddings
        * int2syntax (dictionary): dictionary where keys are integer codes and values are syntax labels
        * sequences (list): list with all possible readings' sequences. The number of sequences depends on the number
        of words with > 1 readings.
        * combinations (DoubleArray): DoubleArray where the first array contains lemmas + tags and the second - only tags.

    Returns:
        * string (string): labelled sentence
    """

    checked_sequences = [elem for elem in sequences]
    checked_sequences = replace_unknown_tags(checked_sequences)

    for i in range(len(sequences)):
        prediction = rnn.generate(
            (checked_sequences[i]),
            vocab,
            int2syntax).split()
        sequence = sequences[i].split()

        for j in range(len(sequence)):
            part = combinations.tags_to_reading(sequence[j])

            if prediction[j] not in part:
                new_part = part + prediction[j]
                string = re.sub(re.escape(part), new_part, string)
                combinations.replace_reading(sequence[j], new_part)

    string = re.sub('<@CLB>', '', string)
    string = re.sub('\^.*?/', '^', string) + '[][]'

    return string


def main():
    input_string = input()
    labelled_sentences = []
    vectors, vocab, int2syntax = prepare_data()

    rnn = SimpleRNNNetwork(2, vectors, 32)
    rnn.model.populate('apertium-language-syntax')

    input_string = re.sub('<@.*?>', '', input_string)
    sentences = split_sentences(input_string)

    for sentence in sentences:
        sequences, combinations = parse_asf(sentence)
        sentence = add_functions(
            rnn,
            sentence,
            vocab,
            int2syntax,
            sequences,
            combinations)
        labelled_sentences.append(sentence)

    output_string = ' '.join(labelled_sentences[:-1]) + labelled_sentences[-1]

    print(output_string)


if __name__ == '__main__':
    main()
