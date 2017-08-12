import re
import sys
import json
import dynet as dy
import numpy as np


def train_test_split(X_corpus, y_corpus): #train/test = 80/20        
    train_ind = round(len(X_corpus) * 0.8)

    train_set = [(X_corpus[i].strip(' \n'), y_corpus[i].strip(' \n')) for i in range (train_ind)]
    test_set = [(X_corpus[i].strip(' \n'), y_corpus[i].strip(' \n')) for i in range (train_ind, len(X_corpus))]
    
    return train_set, test_set


def create_vectors_and_vocab(word2vec_file):
    vocab = {}
    vectors = []
    morph_tags = []

    with open(word2vec_file, 'r', encoding = 'utf-8') as file:
        f = file.read().strip('\n')

    f = re.sub('><', '>!<', f).split('\n')
    
    for i, line in enumerate(f):
        if i != 0:
            word = line.split()
            morph_tags += word[0].split('!')
            word[0] = re.sub('>!<', '><', word[0])
            vocab[word[0]] = i-1
            vectors.append(list(map(float, word[1:])))

    morph_tags = list(set(morph_tags))

    with open('morph_tags.txt', 'w', encoding = 'utf-8') as file:
        file.write(' '.join(morph_tags))

    return vocab, vectors


def create_syntax_stuff(y_corpus):
    syntax = []
    
    for sentence in y_corpus:
        syntax += sentence.split() 

    syntax = list(set(syntax))
    syntax.append("<EOS>")

    syntax2int = {c:i for i,c in enumerate(sorted(syntax))}
    int2syntax = {i:c for i,c in enumerate(sorted(syntax))}

    return syntax2int, int2syntax


def prepare_data(X_filename, y_filename, word2vec_file):
    with open(X_filename, 'r', encoding = 'utf-8') as file:
        X_corpus= file.read().strip('\n\n').split('\n')

    with open(y_filename, 'r', encoding = 'utf-8') as file:
        y_corpus = file.read().strip('\n\n').split('\n')

    train_set, test_set = train_test_split(X_corpus, y_corpus)
    vocab, vectors = create_vectors_and_vocab(word2vec_file)
    syntax2int, int2syntax = create_syntax_stuff(y_corpus)

    return train_set, test_set, vocab, vectors, syntax2int, int2syntax


def train(network, train_set, test_set, epochs = 100):
    def get_val_set_loss(network, test_set):
        loss = [network.get_loss(input_string, output_string).value() for input_string, output_string in test_set]
        return sum(loss)
    
    train_set = train_set*epochs
    trainer = dy.AdadeltaTrainer(network.model)
    c = 0
    
    for i, training_example in enumerate(train_set):
        input_string, output_string = training_example
        
        loss = network.get_loss(input_string, output_string)
        loss_value = loss.value()

        loss.backward()
        trainer.update()

        if i%(len(train_set)/epochs) == 0:
            test_loss = get_val_set_loss(network, test_set)
            test_score = get_accuracy_score(network, test_set)
            train_result = network.generate(train_set[1][0])
            test_result = network.generate(test_set[0][0])

            print('Iteration %s:' % (i/(len(train_set)/epochs)))
            print('Loss on test set: %s' % (test_loss))
            print('Test set accuracy: %s\n' % (test_score))
            print('%s\n%s\n' % (train_set[1][1], train_result))
            print('%s\n%s\n' % (test_set[0][1], test_result))

            if test_score > c:
                filename = 'simple_' + str(test_score)
                c = test_score
                network.model.save(filename)
                print('Saving best model: %s\n' % (test_score))


def get_accuracy_score(network, test_set):
    errors = []

    for i in range (len(test_set)):
        c = 0
        pred_answer = network.generate(test_set[i][0]).split()
        true_answer = test_set[i][1].split()
        
        if len(true_answer) == len(pred_answer):
            for j in range (len(true_answer)):
                if pred_answer[j] == true_answer[j]:
                    c += 1

        errors.append(c / len(true_answer))
    
    return sum(errors) / len(errors)


class SimpleRNNNetwork:
    def __init__(self, rnn_num_of_layers, vectors, state_size):
        self.model = dy.ParameterCollection()
        self.embeddings = self.model.add_lookup_parameters((len(vectors), len(vectors[0])))
        self.embeddings.init_from_array(np.array(vectors))
        self.RNN = dy.LSTMBuilder(rnn_num_of_layers, len(vectors[0]), state_size, self.model)
        self.output_w = self.model.add_parameters((len(vectors), state_size))
        self.output_b = self.model.add_parameters((len(vectors)))
    
    def _preprocess_input(self, string):
        string = string.split() + [EOS]
        return [dy.lookup(self.embeddings, vocab[word]) for word in string]
    
    def _preprocess_output(self, string):
        string = string.split() + [EOS]
        return [syntax2int[c] for c in string]

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

    def get_loss(self, input_string, output_string):
        dy.renew_cg()

        input_string = self._preprocess_input(input_string)
        output_string = self._preprocess_output(output_string)
        
        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, input_string)
        loss = []

        for rnn_output, output_tag in zip(rnn_outputs, output_string):
            probs = self._get_probs(rnn_output)
            loss.append(-dy.log(dy.pick(probs, output_tag)))
        
        loss = dy.esum(loss)
        
        return loss

    def _predict(self, probs):
        probs = probs.value()
        predicted_tag = int2syntax[probs.index(max(probs))]
        
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


X_filename = sys.argv[1]
y_filename = sys.argv[2]
word2vec_file = sys.argv[3]

train_set, test_set, vocab, vectors, syntax2int, int2syntax = prepare_data(X_filename, y_filename, word2vec_file)

f = open('data.json', 'w', encoding='utf-8')
json.dump(int2syntax, f, ensure_ascii = False)
f.close()

EOS = "<EOS>"
RNN_NUM_OF_LAYERS = 2
STATE_SIZE = 32

rnn = SimpleRNNNetwork(RNN_NUM_OF_LAYERS, vectors, STATE_SIZE)
train(rnn, train_set, test_set)