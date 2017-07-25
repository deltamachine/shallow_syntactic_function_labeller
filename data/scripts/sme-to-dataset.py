import re
import sys
import json

def remove_double_lines(corpus):
    with open (corpus, 'r', encoding = 'utf-8') as file:
        corpus = file.read()

    double_analysis = re.findall('\t".*?\n\t".*?\n', corpus)

    for line in double_analysis:
        line = re.sub('\*', '\\*', line)
        line = re.sub('\+', '\\+', line)
        new_line = line.split('\n')[1] + '\n'
        corpus = re.sub(line, new_line, corpus)

    double_words = re.findall('"<.*?\n"<.*?\n', corpus)

    for line in double_words:
        new_line = line.split('\n')[1] + '\n'
        corpus = re.sub(line, new_line, corpus)

    return corpus

def replace_old_tags(corpus):
    with open ('sme-tags.json', 'r', encoding = 'utf-8') as file:
        new_tags = json.load(file)

    for key, value in new_tags.items():
        corpus = re.sub(key, value, corpus)

    corpus = corpus.split('\n\n')

    return corpus

def parse_corpus(corpus, output_morph, output_syntax):
    for elem in corpus:
        sentence = elem.split('\n')
        morph_string = ''
        syntax_string = ''
        
        for i in range (1, len(sentence), 2):
            tags = sentence[i].split()
            
            func_index = len(tags)-1

            for j in range (1, func_index):
                morph_string = morph_string + '<' + tags[j].lower() + '>'
            
            syntax_string = syntax_string + '<' + tags[func_index] + '> '
            morph_string += ' '

        with open(output_morph, 'a', encoding='utf-8') as file:
            file.write('%s%s' % (morph_string, '\n'))

        with open(output_syntax, 'a', encoding='utf-8') as file:
            file.write('%s%s' % (syntax_string, '\n'))

def main():   
    corpus = sys.argv[1]
    output_morph = sys.argv[2]
    output_syntax = sys.argv[3]

    corpus = remove_double_lines(corpus)
    corpus = replace_old_tags(corpus)
    parse_corpus(corpus, output_morph, output_syntax)

if __name__ == '__main__':
    main()
