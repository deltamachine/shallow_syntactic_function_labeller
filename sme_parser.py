import re
import sys


def clean_corpus(input_corpus):
    with open(input_corpus, 'r', encoding='utf-8') as file:
        corpus = file.read()

    corpus = re.sub('CLB', 'CLB @CLB', corpus)
    corpus = re.sub('PUNCT', 'PUNCT @PUNCT', corpus)
    corpus = re.sub('<ext>', 'ext', corpus)
    corpus = re.sub('<hab>', 'hab', corpus)
    corpus = re.sub('<', '←', corpus)
    corpus = re.sub('>', '→', corpus)

    return corpus

def replace_old_tags(corpus):
    new_tags = {' A ': ' Adj ', ' CC ': ' Cnjcoo ', ' CS ': ' Cnjsub ', ' "," CLB ': ' Cm ', \
    ' CLB ': ' Sent ', ' Du1 ': ' P1 Du ', ' Du2 ': ' P2 Du ', ' Du3 ': ' P3 Du ', ' Imprt ': ' Imp ', \
    ' Ind ': ' Indic ', ' Indef ': ' Ind ', ' Interj ': ' Ij ', ' Interr ': ' Itg ', ' PUNCT LEFT ': ' lqout ', \
    ' PUNCT RIGHT ': ' rqout ', ' Pl1 ': ' P1 Pl ', ' Pl2 ': ' P2 Pl ', ' Pl3 ': ' P3 Pl ', ' Po ': ' Post ', \
    ' Pron ': ' Prn ', ' Prs ': ' Pres ', ' Prt ': ' Pret ', ' """ PUNCT ': ' Quot ', ' PUNCT ': ' Quio ', \
    ' PxDu1 ': ' Px1Du ', ' PxDu2 ': ' Px2Du ', ' PxDu3 ': ' Px3Du ', ' PxSg1 ': ' Px1Sg ', ' PxSg2 ': ' Px2Sg ', \
    ' PxSg3 ': ' Px3Sg ', ' PxPl1 ': ' Px1Pl ', ' PxPl2 ': ' Px2Pl ', ' PxPl3 ': ' Px3Pl ', ' RCmpnd ': ' Cmp_SplitR ', \
    ' SgNomCmp ': ' Cmp_SgNom ', ' SgGenCmp ': ' Cmp_SgGen ', ' SgCmp ': ' Cmp_SgNom ', ' Recipr ': ' Res ', \
    ' Refl ': ' Ref ', ' Sg1 ': ' P1 Sg ', ' Sg2 ': ' P2 Sg ', ' P3 ': ' P3 Sg ', ' Sup ': ' Supn ', ' Superl ': ' Sup ', \
    ' V ': ' Vblex '}

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
    input_corpus = sys.argv[1]
    output_morph = sys.argv[2]
    output_syntax = sys.argv[3]

    corpus = clean_corpus(input_corpus)
    corpus = replace_old_tags(corpus)
    
    parse_corpus(corpus, output_morph, output_syntax)

if __name__ == '__main__':
    main()