#!/usr/bin/env python3
#coding=utf-8
# web.py In Memery of Aaron Swartz
# 2017.12.10: Switched into Tornado

import config,utils
import time,json,gc
from action.base import base as BaseAction
from tornado import gen
import tornado.web

class Index(BaseAction):
    def get(self):
        content="Hello, I am a chat bot."
        self.render("publish_index.html",tcontent=content,ptitle="TChatBot")
    
    def post(self):
        x={'a':self.get_argument('a',''), 'cmd':self.get_argument('cmd','')}
        chatbot = self.get_chatbot_instance()
        content = chatbot.get_response(x['cmd'])
        print(content, type(content))
        #content=utils.run(x['cmd'])
        self.render("publish_index.html",tcontent=str(content),ptitle="TChatBot")

class ErrorPage(BaseAction):
    def get(self):
        #self.write("Windows IIS 5.0: Operation not authorized.")
        self.seeother('/')
    
class redirect(BaseAction):
    def get(self, path):
        self.seeother('/' + path)