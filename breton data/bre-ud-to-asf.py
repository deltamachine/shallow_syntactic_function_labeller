import re
import sys


def clean_corpus(input_corpus):
    with open(input_corpus, 'r', encoding='utf-8') as file:
        corpus = file.read().strip('\n')

    corpus = re.sub('#.*?\n', '', corpus)
    corpus = corpus.split('\n\n')

    return corpus


def find_indexes(sentence):
    ind1, ind2 = 100, 100
        
    for i in range (len(sentence)):
        if sentence[i][1] == '-' or (sentence[i][2] == '-' and sentence[i][3] in '0123456789'):
            word = sentence[i].strip('\n').split('\t')
            ind1, ind2 = int(word[0].split('-')[0]), int(word[0].split('-')[1])

    return ind1, ind2


def ud_to_asf(corpus, tags):
    asf_corpus = []

    for sentence in corpus:
        sentence = sentence.split('\n')
        asf_sentence = ''

        ind1, ind2 = find_indexes(sentence)
        
        for i in range (len(sentence)):
            if i < ind1-1 or i > ind2:
                word = sentence[i].strip('\n').split('\t')
                asf_sentence = asf_sentence + '^' + word[2] + '<' + word[4] + '>' + tags[word[5]] + '$ '
                asf_sentence = re.sub('<>', '', asf_sentence)

            if i == ind1-1:
                checker = sentence[i].strip('\n').split('\t')
                word1 = sentence[i+1].strip('\n').split('\t')
                word2 = sentence[i+2].strip('\n').split('\t')
                
                asf_sentence = asf_sentence + '^' + checker[1] + '/' + word1[2] + '<' + word1[4] + '>' + tags[word1[5]] \
                + '+' + word2[2] + '<' + word2[4] + '>' + tags[word2[5]] + '$ '
                asf_sentence = re.sub('<>', '', asf_sentence)

        asf_corpus.append(asf_sentence)
    return asf_corpus


def main():
    input_corpus = sys.argv[1]
    table = sys.argv[2]
    output_corpus = sys.argv[3]

    with open(table, 'r', encoding='utf-8') as file:
        table = file.read().strip('\n').split('\n')

    tags = {elem.split('\t')[0]: elem.split('\t')[1] for elem in table}
    
    corpus = clean_corpus(input_corpus)
    asf_corpus = ud_to_asf(corpus, tags)

    with open (output_corpus, 'w', encoding = 'utf-8') as file:
        for line in asf_corpus:
            file.write('%s%s' % (line, '\n'))


if __name__ == '__main__':
    main()
