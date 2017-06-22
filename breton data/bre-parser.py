import re
import sys


def parse_corpus(input_corpus, tags, punct):
    morph = []
    syntax = []

    with open(input_corpus, 'r', encoding='utf-8') as file:
        corpus = file.readlines()

    for line in corpus:
        if line.strip('\n') == '':
            morph.append(morph_sentence)
            syntax.append(syntax_sentence)
            continue
            
        if line[0] == '#':
            morph_sentence, syntax_sentence = '', ''
            continue

        if line[1] == '-' or (line[2] == '-' and line[3] in '0123456789'):
            continue

        if 'punct' in line and punct == '0':
            continue

        word = line.strip('\n').split('\t')
        syntax_tag = '<@' + word[7] + '>'

        morph_sentence = morph_sentence + '<' + word[4] + '>' + tags[word[5]] + ' '
        morph_sentence = re.sub('<>', '', morph_sentence)

        syntax_sentence = syntax_sentence + syntax_tag + ' '

    return morph, syntax


def save_dataset(data, filename):
    with open (filename, 'w', encoding = 'utf-8') as file:
        for elem in data:
            file.write('%s%s' % (elem, '\n'))


def main():
    input_corpus = sys.argv[1]
    table = sys.argv[2]
    output_morph = sys.argv[3]
    output_syntax = sys.argv[4]
    punct = sys.argv[5]

    with open(table, 'r', encoding='utf-8') as file:
        table = file.read().strip('\n').split('\n')

    tags = {elem.split('\t')[0]: elem.split('\t')[1] for elem in table}

    morph, syntax = parse_corpus(input_corpus, tags, punct)

    save_dataset(morph, output_morph)
    save_dataset(syntax, output_syntax)


if __name__ == '__main__':
    main()