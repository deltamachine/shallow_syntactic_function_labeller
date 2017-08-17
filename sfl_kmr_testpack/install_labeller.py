import os
import re
import sys
from shutil import copyfile


def copy_files(apertium_path):
	list_of_files = ['kmr-eng-labeller.py', 'kmr-embeddings.vec', 'kmr-int2syntax.json', 'kmr-tags.txt', 'kmr-syntax']

	for file in list_of_files:
		copyfile(file, apertium_path + '/' + file)


def change_main_modes(apertium_path, python_path, type_of_change):
	main_modes = apertium_path + '/modes.xml'
	program_change = '<program name="' + python_path + '"/>'
	filename_change = '<file name="kmr-eng-labeller.py"/>'

	with open(main_modes, 'r', encoding = 'utf-8') as file:
		f = file.read()

		if type_of_change == '-cg':
			f = re.sub(program_change, '<program name="apertium-tagger -g $2">',  f)
			f = re.sub(filename_change, '<file name="kmr-eng.prob"/>', f)

		if type_of_change == '-lb':
			f = re.sub(re.escape('<program name="apertium-tagger -g $2">'), program_change, f)
			f = re.sub('<file name="kmr-eng.prob"/>', filename_change, f)

	with open(main_modes, 'w', encoding = 'utf-8') as file:
		file.write(f)


def change_other_modes(apertium_path, python_path, type_of_change):
	list_of_modes = os.listdir(apertium_path + '/modes')
	
	old_part = 'apertium-tagger -g $2 \'' + apertium_path + '/kmr-eng.prob\''
	new_part = python_path + ' \'' + apertium_path + '/kmr-eng-labeller.py\''

	for mode in list_of_modes:
		with open(apertium_path + '/modes/' + mode, 'r', encoding = 'utf-8') as file:
			f = file.read()

			if type_of_change == '-cg':	
				f = re.sub(new_part, old_part, f)
			if type_of_change == '-lb':	
				f = re.sub(re.escape(old_part), new_part, f)

		with open(apertium_path + '/modes/' + mode, 'w', encoding = 'utf-8') as file:
			file.write(f)


def main():
	apertium_path = sys.argv[1]
	python_path = sys.argv[2]
	work_mode = sys.argv[3]
	type_of_change = sys.argv[4]

	if work_mode == '-install':
		copy_files(apertium_path)

	change_main_modes(apertium_path, python_path, type_of_change)
	change_other_modes(apertium_path, python_path, type_of_change)


if __name__ == '__main__':
	main()
