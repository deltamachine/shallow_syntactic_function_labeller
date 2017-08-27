import re
import sys
import json
import dynet as dy
import numpy as np



class AttentionNetwork:
    def __init__(self, enc_layers, dec_layers, vectors, enc_state_size, dec_state_size):
        self.model = dy.Model()
        self.embeddings = self.model.add_lookup_parameters((len(vectors), len(vectors[0])))
        self.embeddings.init_from_array(np.array(vectors))

        self.ENC_RNN = dy.LSTMBuilder(enc_layers, len(vectors[0]), enc_state_size, self.model)
        self.DEC_RNN = dy.LSTMBuilder(dec_layers, enc_state_size, dec_state_size, self.model)

        self.output_w = self.model.add_parameters((len(vectors), dec_state_size))
        self.output_b = self.model.add_parameters((len(vectors)))

        self.attention_w1 = self.model.add_parameters((enc_state_size, enc_state_size))
        self.attention_w2 = self.model.add_parameters((enc_state_size, dec_state_size))
        self.attention_v = self.model.add_parameters((1, enc_state_size))

        self.enc_state_size = enc_state_size

    def _preprocess_input(self, string, vocab):
        string = string.split() + ['<EOS>']
        input_list = []

        for i in range(len(string)):
            string[i] = re.sub('><', '>!<', string[i]).split('!')
            word = np.array([dy.lookup(self.embeddings, vocab[elem])
                             for elem in string[i]])  
            word = np.sum(word, axis=0)

            input_list.append(word)

        return input_list

    def _preprocess_output(self, string, syntax2int):
        string = string.split() + ['<EOS>']
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

    def get_loss(self, input_string, output_string, vocab, syntax2int):
        dy.renew_cg()

        input_string = self._preprocess_input(input_string, vocab)
        output_string = self._preprocess_output(output_string, syntax2int)
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

    def _predict(self, probs, int2syntax):
        probs = probs.value()
        predicted_tag = int2syntax[probs.index(max(probs))]
        
        return predicted_tag

    def generate(self, input_string, vocab, int2syntax):
        dy.renew_cg()

        input_string = self._preprocess_input(input_string, vocab)
        encoded_string = self._encode_string(input_string)
        rnn_state = self.DEC_RNN.initial_state().add_input(dy.vecInput(self.enc_state_size))

        output_string = []
        
        while True:
            attended_encoding = self._attend(encoded_string, rnn_state)
            rnn_state = rnn_state.add_input(attended_encoding)
            probs = self._get_probs(rnn_state.output())
            predicted_tag = self._predict(probs, int2syntax)
            output_string.append(predicted_tag)
            
            if predicted_tag == '<EOS>' or len(output_string) > len(input_string):
                break
        
        output_string = ' '.join(output_string)
        
        return output_string.replace('<EOS>', '')


def train_test_split(X_corpus, y_corpus):
    """ Split corpus with morphological tags' sequences and corpus with syntactic labels' sequences
    in train and test sets in proportion 80/20."""

    train_ind = round(len(X_corpus) * 0.8)

    train_set = [(X_corpus[i].strip(' \n'), y_corpus[i].strip(' \n'))
                 for i in range(train_ind)]

    test_set = [(X_corpus[i].strip(' \n'), y_corpus[i].strip(' \n'))
                for i in range(train_ind, len(X_corpus))]

    return train_set, test_set


def create_vectors_and_vocab(word2vec_file):
    vocab = {}
    vectors = []
    morph_tags = set()

    with open(word2vec_file, 'r', encoding='utf-8') as file:
        f = file.readlines()

    with open('embeddings.vec', 'w', encoding='utf-8') as file:
        for i, line in enumerate(f):
            if i != 0:
                file.write(line + '\n')
                word = line.split()
                morph_tags.add(word[0])
                vocab[word[0]] = i - 1
                vectors.append(list(map(float, word[1:])))

    morph_tags = list(morph_tags)

    with open('tags.txt', 'w', encoding='utf-8') as file:
        file.write(' '.join(morph_tags))

    return vocab, vectors


def create_syntax_stuff(y_corpus):
    syntax = []

    for sentence in y_corpus:
        syntax += sentence.split()

    syntax = list(set(syntax))
    syntax.append("<EOS>")

    syntax2int = {c: i for i, c in enumerate(sorted(syntax))}
    int2syntax = {i: c for i, c in enumerate(sorted(syntax))}

    return syntax2int, int2syntax


def prepare_data(X_filename, y_filename, word2vec_file):
    with open(X_filename, 'r', encoding='utf-8') as file:
        X_corpus = file.read().strip('\n\n').split('\n')

    with open(y_filename, 'r', encoding='utf-8') as file:
        y_corpus = file.read().strip('\n\n').split('\n')

    train_set, test_set = train_test_split(X_corpus, y_corpus)
    vocab, vectors = create_vectors_and_vocab(word2vec_file)
    syntax2int, int2syntax = create_syntax_stuff(y_corpus)

    return train_set, test_set, vocab, vectors, syntax2int, int2syntax


def train(network, train_set, test_set, vocab,
          syntax2int, int2syntax, epochs=100):
    def get_val_set_loss(network, test_set):
        loss = [
            network.get_loss(
                input_string,
                output_string,
                vocab,
                syntax2int).value() for input_string,
            output_string in test_set]

        return sum(loss)

    train_set = train_set * epochs
    trainer = dy.AdadeltaTrainer(network.model)
    c = 0

    for i, training_example in enumerate(train_set):
        input_string, output_string = training_example

        loss = network.get_loss(input_string, output_string, vocab, syntax2int)
        loss_value = loss.value()

        loss.backward()
        trainer.update()

        if i % (len(train_set) / epochs) == 0:
            test_loss = get_val_set_loss(network, test_set)
            test_score = get_accuracy_score(
                network, test_set, vocab, int2syntax)
            train_result = network.generate(train_set[1][0], vocab, int2syntax)
            test_result = network.generate(test_set[0][0], vocab, int2syntax)

            print('Iteration %s:' % (i / (len(train_set) / epochs)))
            print('Loss on test set: %s' % (test_loss))
            print('Test set accuracy: %s\n' % (test_score))
            print('%s\n%s\n' % (train_set[1][1], train_result))
            print('%s\n%s\n' % (test_set[0][1], test_result))

            if test_score > c:
                filename = 'simple_' + str(test_score)
                c = test_score
                network.model.save(filename)
                print('Saving best model: %s\n' % (test_score))


def get_accuracy_score(network, test_set, vocab, int2syntax):
    errors = []

    for i in range(len(test_set)):
        c = 0
        pred_answer = network.generate(
            test_set[i][0], vocab, int2syntax).split()
        true_answer = test_set[i][1].split()

        if len(true_answer) == len(pred_answer):
            for j in range(len(true_answer)):
                if pred_answer[j] == true_answer[j]:
                    c += 1

        errors.append(c / len(true_answer))

    return sum(errors) / len(errors)


def int2syntax_to_json(int2syntax):
    f = open('int2syntax.json', 'w', encoding='utf-8')
    json.dump(int2syntax, f, ensure_ascii=False)
    f.close()


def main():
    X_filename = sys.argv[1]
    y_filename = sys.argv[2]
    word2vec_file = sys.argv[3]

    train_set, test_set, vocab, vectors, syntax2int, int2syntax = prepare_data(
        X_filename, y_filename, word2vec_file)
    
    int2syntax_to_json(int2syntax)

    ENC_RNN_NUM_OF_LAYERS = 2
    DEC_RNN_NUM_OF_LAYERS = 2
    ENC_STATE_SIZE = 32
    DEC_STATE_SIZE = 32


    att = AttentionNetwork(ENC_RNN_NUM_OF_LAYERS, DEC_RNN_NUM_OF_LAYERS, vectors, ENC_STATE_SIZE, DEC_STATE_SIZE)
    train(att, train_set, test_set, vocab, syntax2int, int2syntax)


if __name__ == '__main__':
    main()