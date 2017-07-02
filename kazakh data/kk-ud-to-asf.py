import re
import sys


def clean_corpus(input_corpus):
    with open(input_corpus, 'r', encoding='utf-8') as file:
        corpus = file.read().strip('\n')

    useless_tags = ['ant', 'top', 'hyd', 'cog', 'org', 'al']

    for elem in useless_tags:
        corpus = re.sub(elem, '', corpus)

    corpus = re.sub('NUM\tNUM', 'NUM\tnum', corpus)
    corpus = re.sub('ADP\tADP', 'ADP\tpost', corpus)
    corpus = re.sub('PUNCT\tPUNCT', 'PUNCT\tguio',corpus)
    corpus = re.sub('#.*?\n', '', corpus)

    corpus = corpus.split('\n\n')

    return corpus

def find_tags(word):
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

    return morph_string


def find_indexes(sentence):
    ind1, ind2 = 100, 100
        
    for i in range (len(sentence)):
        if sentence[i][1] == '-' or (sentence[i][2] == '-' and sentence[i][3] in '0123456789'):
            word = sentence[i].strip('\n').split('\t')
            ind1, ind2 = int(word[0].split('-')[0]), int(word[0].split('-')[1])

    return ind1, ind2


def ud_to_asf(corpus):
    asf_corpus = []

    for sentence in corpus:
        sentence = sentence.split('\n')
        asf_sentence = ''

        ind1, ind2 = find_indexes(sentence)
        
        for i in range (len(sentence)):
            if i < ind1-1 or i > ind2:
                word = sentence[i].strip('\n').split('\t')
                morph_string = find_tags(word)
                asf_sentence = asf_sentence + '^' + word[1] + '/' + word[2] + morph_string + '$ '

            if i == ind1-1:
                checker = sentence[i].strip('\n').split('\t')
                if '__' not in checker[1]:
                    word1 = sentence[i+1].strip('\n').split('\t')
                    word2 = sentence[i+2].strip('\n').split('\t')

                    first_morph_string = find_tags(word1)
                    second_morph_string = find_tags(word2)

                    asf_sentence = asf_sentence + '^' + checker[1] + '/' + word1[2] + first_morph_string + '+e' \
                    + second_morph_string + '$ '

                else:
                    if ind2 - ind1 == 2:
                        word1 = sentence[i+1].strip('\n').split('\t')
                        word2 = sentence[i+2].strip('\n').split('\t')
                        word3 = sentence[i+3].strip('\n').split('\t')

                        first_morph_string = find_tags(word1)
                        second_morph_string = find_tags(word2)
                        third_morph_string = find_tags(word3)
                
                        asf_sentence = asf_sentence + '^' + checker[1].split('__')[0] + '/' + word1[2] \
                        + first_morph_string + '+e' + second_morph_string + '$__^' + checker[1].split('__')[1] + '/' + \
                        word3[2] + third_morph_string + '$ '

                    if ind2 - ind1 == 1:
                        word1 = sentence[i+1].strip('\n').split('\t')
                        word2 = sentence[i+2].strip('\n').split('\t')

                        first_morph_string = find_tags(word1)
                        second_morph_string = find_tags(word2)
                        
                        asf_sentence = asf_sentence + '^' + checker[1].split('__')[0] + '/' + word1[2] \
                        + first_morph_string + '$__^' + checker[1].split('__')[1] + '/' + word2[2] + second_morph_string + '$ '

        asf_corpus.append(asf_sentence)                
    return asf_corpus


def main():
    input_corpus = sys.argv[1]
    output_corpus = sys.argv[2]

    corpus = clean_corpus(input_corpus)
    asf_corpus = ud_to_asf(corpus)

    with open (output_corpus, 'w', encoding = 'utf-8') as file:
        for line in asf_corpus:
            file.write('%s%s' % (line, '\n'))

if __name__ == '__main__':
    main()