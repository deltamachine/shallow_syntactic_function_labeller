import sys
import gensim

input_file = sys.argv[1]
output_file = sys.argv[2]

data = gensim.models.word2vec.LineSentence(input_file)
model = gensim.models.Word2Vec(data, size=500, window=10, min_count=1, sg=1)
model.wv.save_word2vec_format(output_file, binary=False)