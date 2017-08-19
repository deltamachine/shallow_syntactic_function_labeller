import os
import re
import sys
from shutil import copyfile
from subprocess import check_output


def find_path(dir_name):
	path = check_output('locate ' + dir_name, shell=True)
	path = str(path).split('\\n')
	path = path[0][2:]

	return path


def copy_files(lang_path, apertium_path):
	list_of_files = os.listdir(lang_path)

	for file in list_of_files:
		copyfile(lang_path + '/' + file, apertium_path + '/' + file)


def change_main_modes(lang_path, python_path, main_changes, work_mode):
	main_modes = lang_path + '/modes.xml'

	with open(main_modes, 'r', encoding = 'utf-8') as file:
		f = file.read()

		if work_mode == '-lb':
			f = re.sub(re.escape(main_changes[0]), main_changes[1],  f)
			f = re.sub(re.escape(main_changes[2]), main_changes[3], f)

		if work_mode == '-cg':
			f = re.sub(re.escape(main_changes[1]), main_changes[0], f)
			f = re.sub(re.escape(main_changes[3]), main_changes[2], f)

	with open(main_modes, 'w', encoding = 'utf-8') as file:
		file.write(f)


def change_other_modes(lang_path, python_path, modes_changes, work_mode):
	list_of_modes = os.listdir(lang_path + '/modes')

	for mode in list_of_modes:
		with open(lang_path + '/modes/' + mode, 'r', encoding = 'utf-8') as file:
			f = file.read()

			if work_mode == '-lb':	
				f = re.sub(re.escape(modes_changes[0]), modes_changes[1], f)
			if work_mode == '-cg':	
				f = re.sub(re.escape(modes_changes[1]), modes_changes[0], f)

		with open(lang_path + '/modes/' + mode, 'w', encoding = 'utf-8') as file:
			file.write(f)


def change_syntax_module(lang_path, python_path, main_changes, modes_changes, work_mode, prefix):
	if work_mode == '-lb':
		module_name = lang_path + '/' + prefix + '-labeller.py'
		
		copy_files(prefix, lang_path)
		copyfile('labeller.py', module_name)

		with open(module_name, 'r', encoding='utf-8')as file:
			f = file.read()

		f = re.sub('apertium-lang', prefix.split('-')[0], f)

		with open(module_name, 'w', encoding='utf-8')as file:
			file.write(f)

	change_main_modes(lang_path, python_path, main_changes, work_mode)
	change_other_modes(lang_path, python_path, modes_changes, work_mode)


def main():
	work_mode = sys.argv[1]
	lang = sys.argv[2]

	sme_path = find_path('apertium-sme-nob')
	kmr_path = find_path('apertium-kmr-eng')
	python_path = sys.executable

	sme_main_changes = ['<program name="cg-proc -1 -n -w" debug-suff="syntax">', '<program name="' + python_path + '" debug-suff="syntax">', '<file name="sme-nob.syn.rlx.bin"/>', '<file name="sme-nob-labeller.py"/>']
	
	sme_modes_changes = ['cg-proc -1 -n -w \'' + sme_path + '/sme-nob.syn.rlx.bin\'', python_path + ' \'' + sme_path + '/sme-nob-labeller.py\'']

	kmr_main_changes = ['<program name="apertium-tagger -g $2">', '<program name="' + python_path + '"/>', '<file name="kmr-eng.prob"/>', '<file name="kmr-eng-labeller.py"/>']
	
	kmr_modes_changes = ['apertium-tagger -g $2 \'' + kmr_path + '/kmr-eng.prob\'', python_path + ' \'' + kmr_path + '/kmr-eng-labeller.py\'']

	if lang == '-sme' or lang == '-all':
		change_syntax_module(sme_path, python_path, sme_main_changes, sme_modes_changes, work_mode, 'sme-nob')
	if lang == '-kmr' or lang == '-all':
		change_syntax_module(kmr_path, python_path, kmr_main_changes, kmr_modes_changes, work_mode, 'kmr-eng')


if __name__ == '__main__':
	main()