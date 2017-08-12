import re
import sys
import json
import dynet as dy
import numpy as np
from itertools import product
from streamparser import parse, readingToString


class SimpleRNNNetwork:
	def __init__(self, rnn_num_of_layers, vectors, state_size):
		self.model = dy.ParameterCollection()
		self.embeddings = self.model.add_lookup_parameters((len(vectors), len(vectors[0])))
		self.embeddings.init_from_array(np.array(vectors))
		self.RNN = dy.LSTMBuilder(rnn_num_of_layers, len(vectors[0]), state_size, self.model)
		self.output_w = self.model.add_parameters((len(vectors), state_size))
		self.output_b = self.model.add_parameters((len(vectors)))
	
	def _preprocess_input(self, string):
		string = string.split() + ['<EOS>']

		return [dy.lookup(self.embeddings, vocab[word]) for word in string]

	def _run_rnn(self, init_state, input_vecs):
		s = init_state
		states = s.add_inputs(input_vecs)
		rnn_outputs = [s.output() for s in states]
		
		return rnn_outputs
	
	def _get_probs(self, rnn_output):
		output_w = dy.parameter(self.output_w)
		output_b = dy.parameter(self.output_b)
		probs = dy.softmax(output_w * rnn_output + output_b)
		
		return probs

	def _predict(self, probs):
		probs = probs.value()
		predicted_tag = int2syntax[str(probs.index(max(probs)))]
		
		return predicted_tag
	
	def generate(self, input_string):
		dy.renew_cg()

		input_string = self._preprocess_input(input_string)

		rnn_state = self.RNN.initial_state()
		rnn_outputs = self._run_rnn(rnn_state, input_string)
		
		output_string = []
		
		for rnn_output in rnn_outputs:
			probs = self._get_probs(rnn_output)
			predicted_tag = self._predict(probs)
			output_string.append(predicted_tag)
		
		output_string = ' '.join(output_string)
		
		return output_string.replace('<EOS>', '')


class DoubleArray:
	def __init__(self, list1, list2):
		self.list1 = list1
		self.list2 = list2

	def add(self, elem1, elem2):
		self.list1.append(elem1)
		self.list2.append(elem2)
	
	def tags_to_reading(self, elem):
		i = self.list2.index(elem)
		return self.list1[i]
	
	def replace_reading(self, elem, new_elem):
		i = self.list2.index(elem)
		self.list1[i] = new_elem


def prepare_data():
	vectors =  []
	vocab = {}

	with open('br-embeddings.vec', 'r', encoding = 'utf-8') as file:
		f = file.readlines()

	with open('br-int2syntax.json', 'r', encoding = 'utf-8') as jsonfile:
		s = jsonfile.read()
		
	for i, line in enumerate(f):
		word = line.split()
		vocab[word[0]] = i
		vectors.append(list(map(float, word[1:])))

	int2syntax = json.loads(s)

	return vectors, vocab, int2syntax


def parse_asf(string):
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

	elements = [options[key.strip(' ^')] for key in string.split('$')[:-1]]
	elements = product(*elements)
	sequences = [' '.join(elem) for elem in elements]

	return sequences, combinations


def replace_useless_tags(sequences):
	for i in range(len(sequences)):
		sequences[i] = re.sub('<ABBR>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<ACR>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<Allegro>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<G3>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<G7>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<ext>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<Foc_>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<Qst>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<sem_state>', '', sequences[i], flags = re.IGNORECASE)
		sequences[i] = re.sub('<sp>', '', sequences[i], flags = re.IGNORECASE)
	return sequences


def add_functions(rnn, string, sequences, checked_sequences, combinations):
	for i in range(len(sequences)):
		prediction = rnn.generate(checked_sequences[i]).split()
		sequence = sequences[i].split()
		
		for j in range(len(sequence)):
			part = combinations.tags_to_reading(sequence[j])
			
			if prediction[j] not in part:
				new_part = part + prediction[j]
				string = re.sub(part, new_part, string)
				combinations.replace_reading(sequence[j], new_part)

	string = re.sub('<@CLB>', '', string)
	
	return string


vectors, vocab, int2syntax = prepare_data()

rnn = SimpleRNNNetwork(2, vectors, 32)
rnn.model.populate('br-syntax')

original_string = '^Er/E<pr>+an<det><def><sp>$ ^sal/sal<n><f><sg>$ ^dour/dour<n><m><sg>/tour<n><m><sg>/dourañ<vblex><pri><p3><sg>/dourañ<vblex><imp><p2><sg>$ ^en em/en em<vpart><ref>$ ^walc\'hen/gwalc’hañ<vblex><pii><p1><sg>/gwalc’hiñ<vblex><pii><p1><sg>$^./.<sent>$'
undisambiguated_string = re.sub('<@.*?>', '', original_string)
sequences, combinations = parse_asf(undisambiguated_string)
checked_sequences = [elem for elem in sequences]
checked_sequences= replace_useless_tags(checked_sequences)
string = add_functions(rnn, original_string, sequences, checked_sequences, combinations)

print(string)