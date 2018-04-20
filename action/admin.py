#!/usr/bin/env python3
#coding=utf-8
# web.py In Memery of Aaron Swartz
# 2017.12.10: Switched into Tornado

import config,utils
import time,json,gc
from action.base import base as BaseAction
from action.base import wsbase as WSBaseAction
from tornado import gen
import tornado.web
import tornado.websocket

class Index(BaseAction):
    def get(self):
        content="Hello, I am a chat bot."
        self.render("publish_index.html",tcontent=content, netConf=config.netpref, ptitle="TChatBot")
    
    def post(self):
        x={'a':self.get_argument('a',''), 'cmd':self.get_argument('cmd','')}
        chatbot = self.get_chatbot_instance()
        content = chatbot.get_response(x['cmd'])
        print(content, type(content))
        self.render("publish_index.html",tcontent=str(content), netConf=config.netpref, ptitle="TChatBot")

class ChatWebSocket(WSBaseAction):
    def open(self):
        print("WebSocket opened")
        chatbot = self.get_chatbot_instance()
        self.write_message(chatbot.name)

    def on_message(self, message):
        chatbot = self.get_chatbot_instance()
        content = chatbot.get_response(message)
        self.write_message(str(content))

    def on_close(self):
        print("WebSocket closed")


class ErrorPage(BaseAction):
    def get(self):
        #self.write("Windows IIS 5.0: Operation not authorized.")
        self.seeother('/')
    
class redirect(BaseAction):
    def get(self, path):
        self.seeother('/' + path)