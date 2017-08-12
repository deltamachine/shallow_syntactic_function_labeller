import re
import sys
import json
import dynet as dy
import numpy as np
from streamparser import parse


def parse_asf(string):
	units = parse(c for c in string)
	tag_string = []
	all_tags = []

	for word in units:
		word = str(word) + '$'
		tags = re.findall('[^>](<.*?>)[\+/\$]', word)
		all_tags += tags
		
		if tags == []:
			tag_string.append('<unknown>')
		else:
			tag_string += [tag for tag in tags]

	tag_string = ' '.join(tag_string)

	return tag_string, all_tags


def add_functions(original, functions, tags):
	functions = functions.split()
	string = original.split()
	output_string = ''
	j = 0
	
	if '+' not in original:
		for i in range(len(string)):
			string[i] = re.sub(tags[i], tags[i] + functions[i], string[i])
			output_string = output_string + string[i] + ' '
	else:
		for i in range(len(string)):
			if '+' not in string[i]:
				print(string[i])
				string[i] = re.sub(tags[j], tags[j] + functions[j], string[i])
				output_string = output_string + string[i] + ' '
				j +=1
			else:
				compound = string[i].split('+')
				for k in range(len(compound)):
					compound[k] = re.sub(tags[j], tags[j] + functions[j], compound[k])
					j +=1
				
				compound = '+'.join(compound)
				output_string = output_string + compound + ' '

	return output_string.strip(' ')


def prepare_data():
	vectors =  []
	vocab = {}

	with open('embeddings.vec', 'r', encoding = 'utf-8') as file:
		f = file.readlines()

	with open('int2syntax.json', 'r', encoding = 'utf-8') as jsonfile:
		s = jsonfile.read()
		
	for i, line in enumerate(f):
		word = line.split()
		vocab[word[0]] = i
		vectors.append(list(map(float, word[1:])))

	int2syntax = json.loads(s)

	return vectors, vocab, int2syntax


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


vectors, vocab, int2syntax = prepare_data()

rnn = SimpleRNNNetwork(2, vectors, 32)
rnn.model.populate('syntax')

#string = '^ret<adj>$ ^bezañ<vblex><pri><p3><sg>$ ^dit/da<pr>+indirect<prn><obj><p2><sg>$ ^dont<vblex><inf>$ ^.<sent>$'

string = sys.argv[1]

tag_string, all_tags = parse_asf(string)
print(tag_string)
answer = rnn.generate(tag_string)
print(answer)

last = add_functions(string, answer, all_tags)
print(last)