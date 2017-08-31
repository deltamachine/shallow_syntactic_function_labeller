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
from streamparser import parse


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


def handle_input(input_string):
    """ Process input string: delete disambiguation tags, handle unknown words, split string into sentences. """

    input_string = re.sub('<@.*?>', '', input_string)
    unknown_words = re.findall('\*.*?\$', input_string)

    for elem in unknown_words:
        input_string = re.sub(
            re.escape(elem),
            elem.strip('$') + '<UNK>$',
            input_string)

    sentences = split_sentences(input_string)

    return sentences


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


def parse_asf(input_string):
    """ Take a string in Apertium stream format and parse it into a sequence (or into a few possible sequences)
    of morphological tags using streamparser library.

    Returns:
        * sequences (list): list with all possible readings' sequences. The number of sequences depends on the number
        of words with > 1 readings.
    """

    units = parse(c for c in input_string)
    options = {}

    for unit in units:
        all_tags = []

        for reading in unit.readings:
            joined_lexical_unit_tags = ['<' + '><'.join(reading[i][1]) + '>' for i in range(len(reading))]
            all_tags.append(' '.join(joined_lexical_unit_tags))

        options[str(unit)] = all_tags

    splitted_string = input_string.split('$')[:-1]
    combinations = [options[re.sub('.*?\^', '', key)]
                    for key in splitted_string]
    
    combinations = product(*combinations)
    sequences = {' '.join(elem): [] for elem in combinations}

    return sequences


def replace_unknown_tags(checked_sequences):
    """ Delete all tags in sequences which model doesn't know """

    with open('apertium-language-tags.txt', 'r', encoding='utf-8') as file:
        morph_tags = file.read().split()

    for i in range(len(checked_sequences)):
        sequence_tags = checked_sequences[i][0].split()

        for j in range(len(sequence_tags)):
            sequence_tags[j] = re.sub('><', '>!<', sequence_tags[j]).split('!')

            for elem in sequence_tags[j]:
                if elem not in morph_tags:
                    checked_sequences[i][0] = re.sub(
                        elem, '', checked_sequences[i][0])

    return checked_sequences


def prepare_data_for_labelling(string, sequences):
    """ Split string in a needed way, create an array with all possible sequences
    and deletes all unknown to pretrained model tags from these sequences. """

    string = re.sub('\$\^', '$ ^', string)
    string = re.sub('\$ \^', '$&^', string).split('&')

    checked_sequences = [[key, key] for key in sequences.keys()]
    checked_sequences = replace_unknown_tags(checked_sequences)

    return string, checked_sequences


def get_predictions(rnn, sequences, checked_sequences, vocab, int2syntax):
    """ Just get a prediction for every possible sequence. """

    for i in range(len(checked_sequences)):
        prediction = rnn.generate(
            (checked_sequences[i][0]),
            vocab,
            int2syntax).split()
        sequences[checked_sequences[i][1]] = prediction

    return sequences


def get_a_label(part_of_string, part_of_sequence, part_of_prediction):
    """ Well, some insane shit is going on here, I don't even want to comment it.
    But, at least, it seems to solve words-without-a-label and wrong-tags-order bugs.
    (If I find more problems, I'll probably kill myself) """

    readings = re.sub('/', '/&', part_of_string).split('&')[1:]

    for reading in readings:
        tags_and_separator = re.search('(<.*>)(/|\$)', reading)
        all_tags = tags_and_separator.group(1)
        separator = tags_and_separator.group(2)
        morph_tags = all_tags.split('<@')[0]

        if part_of_sequence == morph_tags and part_of_prediction not in all_tags:
            old_part = re.escape(all_tags + separator)
            new_part = all_tags + part_of_prediction + separator
            part_of_string = re.sub(old_part, new_part, part_of_string)

    return part_of_string


def prepare_output(string):
    """ Format labelled string in a needed way. """

    string = ' '.join(string[:-1]) + string[-1]
    string = re.sub('<@CLB>', '', string)
    string = re.sub('<UNK>', '', string)
    string = re.sub('\^.*?/', '^', string) + '[][]'

    return string


def add_functions(rnn, string, vocab, int2syntax, sequences):
    """ Add syntactic function labels to a sentence from the original string.

    Args:
        * rnn (SimpleRNNNetwork): pretrained RNN Network
        * string (string): sequence of morpholodical tags (one sentence)
        * vocab (dictionary): dictionary where keys are tags and values are embeddings
        * int2syntax (dictionary): dictionary where keys are integer codes and values are syntax labels
        * sequences (list): list with all possible readings' sequences. The number of sequences depends on the number
        of words with > 1 readings.

    Returns:
        * string (string): labelled sentence
    """

    string, checked_sequences = prepare_data_for_labelling(string, sequences)
    sequences = get_predictions(
        rnn,
        sequences,
        checked_sequences,
        vocab,
        int2syntax)

    for variant, prediction in sequences.items():
        variant = variant.split()

        for i in range(len(variant)):
            string[i] = get_a_label(string[i], variant[i], prediction[i])

    string = prepare_output(string)

    return string


def main():
    input_string = input()
    labelled_sentences = []
    vectors, vocab, int2syntax = prepare_data()

    rnn = SimpleRNNNetwork(2, vectors, 32)
    rnn.model.populate('apertium-language-syntax')

    sentences = handle_input(input_string)

    for sentence in sentences:
        sequences = parse_asf(sentence)
        sentence = add_functions(
            rnn,
            sentence,
            vocab,
            int2syntax,
            sequences)
        labelled_sentences.append(sentence)

    output_string = ' '.join(labelled_sentences[:-1]) + labelled_sentences[-1]

    print(output_string)


if __name__ == '__main__':
    main()
