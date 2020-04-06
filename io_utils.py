import yaml
import json

def load_yaml(filepath):
	'''
	Load yaml file provided a filepath
	'''
	with open(filepath, 'r') as fin:
		data = yaml.load(fin)
	return data

def save_yaml(data, filepath):
	'''
	Saves python dictionary to a yaml file
	'''
	with open(filepath, 'w') as fout:
		yaml.dump(data, fout)


def load_json(filepath):
	'''
	Load json file provided a filepath
	'''
	with open(filepath, 'r') as fin:
		data = json.load(fin)
	return data


def save_json(data, filepath):
	'''
	Saves python dictionary to a json file
	'''

	with open(filepath, 'w') as fout:
		json.dump(data, fout)