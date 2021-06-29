from parsing import * 
import json
import os

def main():
	with open('/Users/asifanwar/Dev/git/TinyML/test/onejsonfile.json') as json_file:
		data = json.load(json_file)

if __name__ == '__main__':
	main()