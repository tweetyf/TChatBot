#!/usr/bin/env python3
#coding=utf-8
'''This is a standalone tool to check unusual data.'''

from pymongo import MongoClient
from pymongo.errors import OperationFailure

database_name = 'tchatterbot-db'
localurl = 'mongodb://localhost:27017/'
client = MongoClient(localurl)

database = client[database_name]
# The mongo collection of statement documents
statements = database['statements']
# The mongo collection of conversation documents
conversations = database['conversations']

cursor = statements.find()

for cr in cursor:
	if (len(cr['in_response_to']) > 50):
		print(type(cr), len(cr), len(cr['in_response_to']), cr['text'], str(len(str(cr))/1024)+'kb' )
	continue;