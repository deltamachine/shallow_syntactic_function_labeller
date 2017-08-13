import os
import re
import sys


def change_main_modes(apertium_path, python_path, type_of_change):
	main_modes = apertium_path + '/modes.xml'
	program_change = '<program name="' + python_path + '" debug-suff="syntax">'
	filename_change = '<file name="sme-nob-labeller.py"/>'

	with open(main_modes, 'r', encoding = 'utf-8') as file:
		f = file.read()

		if type_of_change == 'cg':
			f = re.sub(program_change, '<program name="cg-proc -1 -n -w" debug-suff="syntax">',  f)
			f = re.sub(filename_change, '<file name="sme-nob.syn.rlx.bin"/>', f)

		if type_of_change == 'lb':
			f = re.sub('<program name="cg-proc -1 -n -w" debug-suff="syntax">', program_change, f)
			f = re.sub('<file name="sme-nob.syn.rlx.bin"/>', filename_change, f)

	with open(main_modes, 'w', encoding = 'utf-8') as file:
		file.write(f)


def change_other_modes(apertium_path, python_path, type_of_change):
	list_of_modes = os.listdir(apertium_path + '/modes')
	
	old_part = 'cg-proc -1 -n -w \'' + apertium_path + '/sme-nob.syn.rlx.bin\''
	new_part = python_path + ' \'' + apertium_path + '/sme-nob-labeller.py\''

	for mode in list_of_modes:
		with open(apertium_path + '/modes/' + mode, 'r', encoding = 'utf-8') as file:
			f = file.read()

			if type_of_change == 'cg':	
				f = re.sub(new_part, old_part, f)
			if type_of_change == 'lb':	
				f = re.sub(old_part, new_part, f)

		with open(apertium_path + '/modes/' + mode, 'w', encoding = 'utf-8') as file:
			file.write(f)


def main():
	apertium_path = sys.argv[1]
	python_path = sys.argv[2]
	type_of_change = sys.argv[3]

	change_main_modes(apertium_path, python_path, type_of_change)
	change_other_modes(apertium_path, python_path, type_of_change)


if __name__ == '__main__':
	main()