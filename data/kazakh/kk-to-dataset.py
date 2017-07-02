import re
import sys


def clean_corpus(input_corpus):
    with open(input_corpus, 'r', encoding='utf-8') as file:
        corpus = file.read()

    useless_tags = ['ant', 'top', 'hyd', 'cog', 'org', 'al']

    for elem in useless_tags:
        corpus = re.sub(elem, '', corpus)

    corpus = re.sub('NUM\tNUM', 'NUM\tnum', corpus)
    corpus = re.sub('ADP\tADP', 'ADP\tpost', corpus)
    corpus = re.sub('PUNCT\tPUNCT', 'PUNCT\tguio',corpus)

    corpus = corpus.split('\n')

    return corpus

def parse_corpus(corpus, punct):
    morph = []
    syntax = []

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

        if 'PUNCT' in line and punct == '0':
            continue

        word = line.strip('\n').split('\t')
        tags = word[9].split('|')
        morph_string = ''

        if '|' + word[4] in word[9]:
            for i in range (1, len(tags)):
                new_tag = '<' + tags[i] + '>'
                morph_string += new_tag

        else:
            pos_tag = '<' + word[4] + '>'
            morph_string += pos_tag
            
            if '_' not in tags:
                for tag in tags:
                    new_tag = '<' + tag + '>'
                    morph_string += new_tag

        syntax_tag = '<@' + word[7] + '>'

        morph_sentence = morph_sentence + morph_string + ' '
        syntax_sentence = syntax_sentence + syntax_tag + ' '

    return morph, syntax

def save_dataset(data, filename):
    with open (filename, 'w', encoding = 'utf-8') as file:
        for elem in data:
            file.write('%s%s' % (elem, '\n'))

def main():
    input_corpus = sys.argv[1]
    output_morph = sys.argv[2]
    output_syntax = sys.argv[3]
    punct = sys.argv[4]

    corpus = clean_corpus(input_corpus)
    morph, syntax = parse_corpus(corpus, punct)

    save_dataset(morph, output_morph)
    save_dataset(syntax, output_syntax)

if __name__ == '__main__':
    main()