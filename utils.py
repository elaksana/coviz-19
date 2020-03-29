import json

def load_json(filepath):
	'''
	Load json file provided a filepath
	'''
	with open(filepath, 'r') as f:
		data = json.load(f)
	return data


def save_json(data, filepath):
	'''
	Saves python dictionary to a json file
	'''

	with open(filepath, 'w') as json_file:
		json.dump(data, filepath)