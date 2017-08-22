import os
import re
import sys
from subprocess import check_output

input_file = re.escape(sys.argv[1])
output_file = re.escape(sys.argv[2])

path = check_output('locate fastText', shell=True)
path = str(path).split('\\n')
path = path[0][2:]

string = './fasttext skipgram -input ' + input_file + ' -output ' + output_file + ' -minCount 1 -dim 300 -maxn 11'

os.chdir(path)
os.system(string)