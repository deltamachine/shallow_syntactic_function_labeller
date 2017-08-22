import re
import sys


def read_tags(lang):
    if lang == 'kk' or lang == 'kmr':
        return {}
    else:
        if lang == 'bre':
            tablename = 'bre-tags.tsv'
        if lang == 'en':
            tablename = 'en-tags.tsv'

    with open(tablename, 'r', encoding='utf-8') as file:
        table = file.read().strip('\n').split('\n') 
        
    tags = {elem.split('\t')[0]: elem.split('\t')[1] for elem in table}

    return tags


def clean_corpus(input_corpus, lang):
    with open(input_corpus, 'r', encoding='utf-8') as file:
        corpus = file.read()

    if lang == 'kk':
        useless_tags = ['ant', 'top', 'hyd', 'cog', 'org', 'al']

        for elem in useless_tags:
            corpus = re.sub(elem, '', corpus)

        corpus = re.sub('NUM\tNUM', 'NUM\tnum', corpus)
        corpus = re.sub('ADP\tADP', 'ADP\tpost', corpus)
        corpus = re.sub('PUNCT\tPUNCT', 'PUNCT\tguio',corpus)

    if lang == 'en':
        corpus = re.sub('\tPUNCT.*?\n\n', '\tsent\t\.\t_\t_\tpunct\t_\t_\n\n', corpus)
        corpus = re.sub(',\tPUNCT', ',\tcm', corpus)
        corpus = re.sub('\tPUNCT', '\tguio', corpus)

    corpus = corpus.split('\n')

    return corpus


def handle_breton(word, tags):   
    morph_string = '<' + word[4] + '>' + tags[word[5]] + ' '
    morph_string = re.sub('<>', '', morph_string)

    return morph_string


def handle_english(word, tags):
    morph_string = tags[word[3]] + tags[word[5]] + ' '
    morph_string = re.sub('<>', '', morph_string)

    return morph_string


def handle_kurmanji(word):
    if word[5] == '_':
        morph_string = '<' + word[4] + '> '
    else:
        morph_string = '<' + word[4] + '><' + '><'.join(word[5].split('|')) + '> '
        
    return morph_string


def handle_kazakh(word):
    morph_string = ''
    tags = word[9].split('|')

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

    morph_string += ' '

    return morph_string


def process_line(line, lang, tags):
    word = line.strip('\n').split('\t')

    if lang == 'bre':
        morph_string = handle_breton(word, tags)

    if lang == 'en':
        morph_string = handle_english(word, tags)

    if lang == 'kmr':
        morph_string = handle_kurmanji(word)

    if lang == 'kk':
        morph_string = handle_kazakh(word)

    syntax_tag = '<@' + word[7] + '> '

    return morph_string, syntax_tag


def parse_corpus(corpus, tags, lang, punct):
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

        if ('PUNCT' in line or 'punct' in line) and punct == '0':
            continue

        morph_string, syntax_tag = process_line(line, lang, tags)

        morph_sentence += morph_string
        syntax_sentence += syntax_tag

    return morph, syntax


def save_dataset(data, filename):
    with open (filename, 'w', encoding = 'utf-8') as file:
        for elem in data:
            file.write('%s%s' % (elem, '\n'))


def main():
    input_corpus = sys.argv[1]
    output_morph = sys.argv[2]
    output_syntax = sys.argv[3]
    lang = sys.argv[4]
    punct = sys.argv[5]

    tags = read_tags(lang)
    corpus = clean_corpus(input_corpus, lang)
    morph, syntax = parse_corpus(corpus, tags, lang, punct)

    save_dataset(morph, output_morph)
    save_dataset(syntax, output_syntax)


if __name__ == '__main__':
    main()
