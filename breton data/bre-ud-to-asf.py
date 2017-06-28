import re
import sys


def ud_to_asf(input_corpus, tags):
    asf_corpus = []

    with open(input_corpus, 'r', encoding='utf-8') as file:
        corpus = file.readlines()

    for line in corpus:
        if line.strip('\n') == '':
            asf_corpus.append(asf_sentence)
            continue
            
        if line[0] == '#':
            asf_sentence = ''
            continue

        if line[1] == '-' or (line[2] == '-' and line[3] in '0123456789'):
            continue

        word = line.strip('\n').split('\t')
        asf_sentence = asf_sentence + '^' + word[2] + '<' + word[4] + '>' + tags[word[5]] + '$ '
        asf_sentence = re.sub('<>', '', asf_sentence)

    return asf_corpus


def main():
    input_corpus = sys.argv[1]
    table = sys.argv[2]
    output_corpus = sys.argv[3]

    with open(table, 'r', encoding='utf-8') as file:
        table = file.read().strip('\n').split('\n')

    tags = {elem.split('\t')[0]: elem.split('\t')[1] for elem in table}
    asf_corpus = ud_to_asf(input_corpus, tags)

    with open (output_corpus, 'w', encoding = 'utf-8') as file:
        for line in asf_corpus:
            file.write('%s%s' % (line, '\n'))


if __name__ == '__main__':
    main()
