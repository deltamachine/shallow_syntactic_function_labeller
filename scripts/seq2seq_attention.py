import sys
import dynet as dy
import numpy as np


def needed_data(y_corpus, word2vec_file):
    syntax = []
    vocab = {}
    vectors = []
    
    for i in range(len(y_corpus)):
        syntax += y_corpus[i].split()
    
    syntax = list(set(syntax))

    with open(word2vec_file, 'r', encoding = 'utf-8') as file:
        f = file.readlines()
    
    for i, line in enumerate(f):
        if i != 0:
            word = line.split()
            vocab[word[0]] = i-1
            vectors.append(list(map(float, word[1:])))

    return vocab, vectors, syntax


def train_test_split(X_corpus, y_corpus): #train/test = 80/20        
    train_ind = round(len(X_corpus) * 0.6)
    train_set = []
    test_set = []
    
    for i in range(train_ind):
        train_set.append((X_corpus[i].strip(' \n'), y_corpus[i].strip(' \n')))
    
    for i in range(train_ind, len(X_corpus)):
        test_set.append((X_corpus[i].strip(' \n'), y_corpus[i].strip(' \n')))
    
    return train_set, test_set


def prepare_data(X_filename, y_filename, word2vec_file):
    EOS = "<EOS>"

    with open(X_filename, 'r', encoding = 'utf-8') as file:
        X_corpus= file.read().strip('\n\n').split('\n')

    with open(y_filename, 'r', encoding = 'utf-8') as file:
        y_corpus = file.read().strip('\n\n').split('\n')

    train_set, test_set = train_test_split(X_corpus, y_corpus)
    vocab, vectors, syntax = needed_data(y_corpus, word2vec_file)

    syntax.append(EOS)

    syntax2int = {c:i for i,c in enumerate(syntax)}
    int2syntax = {i:c for i,c in enumerate(syntax)}
    
    return train_set, test_set, vocab, vectors, syntax2int, int2syntax


def train(network, train_set, test_set, iterations = 50):
    def get_val_set_loss(network, test_set):
        loss = [network.get_loss(input_string, output_string).value() for input_string, output_string in test_set]
        return sum(loss)
    
    train_set = train_set*iterations 
    trainer = dy.SimpleSGDTrainer(network.model)
    c = 0
    
    for i, training_example in enumerate(train_set):
        input_string, output_string = training_example
        
        loss = network.get_loss(input_string, output_string)
        loss_value = loss.value()

        loss.backward()
        trainer.update()

        if i%(len(train_set)/iterations) == 0:
            test_loss = get_val_set_loss(network, test_set)
            test_score = get_accuracy_score(network, test_set)
            train_result = network.generate(train_set[0][0])
            test_result = network.generate(test_set[0][0])

            print('Iteration %s:' % (i/(len(train_set)/iterations)))
            print('Loss on test set: %s' % (test_loss))
            print('Test set accuracy: %s\n' % (test_score))
            print('%s\n%s\n' % (train_set[0][1], train_result))
            print('%s\n%s\n' % (test_set[0][1], test_result))

            if test_score > c:
                filename = 'att_' + str(test_score)
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
    accuracy_score = sum(errors) / len(errors)
    
    return accuracy_score

class AttentionNetwork:
    def __init__(self, enc_layers, dec_layers, vectors, enc_state_size, dec_state_size):
        self.model = dy.Model()
        self.embeddings = self.model.add_lookup_parameters((len(vectors), len(vectors[0])))
        self.embeddings.init_from_array(np.array(vectors))

        self.ENC_RNN = RNN_BUILDER(enc_layers, len(vectors[0]), enc_state_size, self.model)
        self.DEC_RNN = RNN_BUILDER(dec_layers, enc_state_size, dec_state_size, self.model)

        self.output_w = self.model.add_parameters((len(vectors), dec_state_size))
        self.output_b = self.model.add_parameters((len(vectors)))

        self.attention_w1 = self.model.add_parameters((enc_state_size, enc_state_size))
        self.attention_w2 = self.model.add_parameters((enc_state_size, dec_state_size))
        self.attention_v = self.model.add_parameters((1, enc_state_size))

        self.enc_state_size = enc_state_size

    def _preprocess_input(self, string):
        string = string.split() + [EOS]
        return [dy.lookup(self.embeddings, vocab[word]) for word in string]
    
    def _preprocess_output(self, string):
        string = string.split() + [EOS]
        return [syntax2int[c] for c in string]

    def _encode_string(self, input_string):
        initial_state = self.ENC_RNN.initial_state()
        hidden_states = self._run_rnn(initial_state, input_string)

        return hidden_states

    def _attend(self, input_vectors, state):
        w1 = dy.parameter(self.attention_w1)
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        
        attention_weights = []
        w2dt = w2 * state.h()[-1]

        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)
        
        attention_weights = dy.softmax(dy.concatenate(attention_weights))
        output_vectors = dy.esum([vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
        
        return output_vectors

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
        encoded_string = self._encode_string(input_string)

        rnn_state = self.DEC_RNN.initial_state().add_input(dy.vecInput(self.enc_state_size))
        loss = []

        for output_tag in output_string:
            attended_encoding = self._attend(encoded_string, rnn_state)
            rnn_state = rnn_state.add_input(attended_encoding)
            probs = self._get_probs(rnn_state.output())
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
        encoded_string = self._encode_string(input_string)
        rnn_state = self.DEC_RNN.initial_state().add_input(dy.vecInput(self.enc_state_size))

        output_string = []
        
        while True:
            attended_encoding = self._attend(encoded_string, rnn_state)
            rnn_state = rnn_state.add_input(attended_encoding)
            probs = self._get_probs(rnn_state.output())
            predicted_tag = self._predict(probs)
            output_string.append(predicted_tag)
            
            if predicted_tag == EOS or len(output_string) > len(input_string):
                break
        
        output_string = ' '.join(output_string)
        
        return output_string.replace('<EOS>', '')


X_filename = sys.argv[1]
y_filename = sys.argv[2]
word2vec_file = sys.argv[3]

train_set, test_set, vocab, vectors, syntax2int, int2syntax = prepare_data(X_filename, y_filename, word2vec_file)

RNN_BUILDER = dy.LSTMBuilder
EOS = "<EOS>"
ENC_RNN_NUM_OF_LAYERS = 2
DEC_RNN_NUM_OF_LAYERS = 2
ENC_STATE_SIZE = 32
DEC_STATE_SIZE = 32


att = AttentionNetwork(ENC_RNN_NUM_OF_LAYERS, DEC_RNN_NUM_OF_LAYERS, vectors, ENC_STATE_SIZE, DEC_STATE_SIZE)
train(att, train_set, test_set)
