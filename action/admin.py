#!/usr/bin/env python3
#coding=utf-8
# web.py In Memery of Aaron Swartz
# 2017.12.10: Switched into Tornado

import config,utils
import time,json,gc,random,re
from action.base import base as BaseAction
from action.base import wsbase as WSBaseAction
from tornado import gen
import tornado.web
import tornado.websocket

rAtSm = re.compile("\@-?[1-9]\d* ",re.DOTALL)

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
        result ={}
        result['id']= chatbot.name
        result['m']='您好，我是'+chatbot.name
        #print(json.dumps(result))
        self.write_message(json.dumps(result))

    def on_message(self, message):
        chatbot = self.get_chatbot_instance()
        # make first response.
        result ={}
        msg = json.loads(message)
        convID = msg.get('id')
        convMsg = msg.get('m')
        # process the message
        convMsg = convMsg[0:100]
        aaa = rAtSm.findall(convMsg)
        for a in aaa:
            convMsg = convMsg.replace(a,'')
        # message process end.
        # make a unique Conversation ID, aproxmate in 100s.
        unicConvID = convID + ':' + str(time.time())[0:8]
        print('Session iD:' ,unicConvID)
        # get response from server
        content = chatbot.get_response(convMsg,unicConvID)
        result['id']= convID
        result['m']=str(content)
        self.write_message(json.dumps(result))
        if (random.randint(0,100) <4):
            # 8% chance to make 3rd response.
            content = chatbot.get_response(str(content),unicConvID)
            result['m']=str(content)
            self.write_message(json.dumps(result))

    def on_close(self):
        print("WebSocket closed")


class ErrorPage(BaseAction):
    def get(self):
        #self.write("Windows IIS 5.0: Operation not authorized.")
        self.seeother('/')
    
class redirect(BaseAction):
    def get(self, path):
        self.seeother('/' + path)
