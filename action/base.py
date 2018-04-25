#!/usr/bin/env python3
#coding=utf-8
# web.py In Memery of Aaron Swartz
# 2017.12.10: Switched into Tornado

import tornado.web
import tornado.websocket
import config,utils
import json,time,base64,urllib.request,urllib.error,urllib.parse
import re
import os
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

class MyChatBot(ChatBot):
    def nothing(self):
        pass

# Create a new instance of a ChatBot
Schatbot = MyChatBot(
    config.DEFAULT_BOTNAME,
    #storage_adapter="chatterbot.storage.SQLStorageAdapter",
    #database="./chatterbot.database.db",
    storage_adapter='model.MongoDatabaseAdapter',
    database='tchatterbot-db',
    logic_adapters=[
        #"chatterbot.logic.BestMatch",
        {
            "import_path": "logic.BestMatch",
            "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
            "response_selection_method": "chatterbot.response_selection.get_random_response"
        },
    #    "chatterbot.logic.MathematicalEvaluation",
    #    "chatterbot.logic.TimeLogicAdapter",
    ],
    output_adapter="chatterbot.output.OutputAdapter",
    output_format="text",
    read_only=False,
)

class wsbase(tornado.websocket.WebSocketHandler):
    def get_chatbot_instance(self):
        return Schatbot

class base(tornado.web.RequestHandler):

    ### Templates
    t_globals = {
        'strdate':utils.strtime,
        'inttime':utils.inttime,
        'abs2rev':utils.abs2rev
    }
    def get_chatbot_instance(self):
        return Schatbot

    def get_current_user(self):
        user_id =  self.get_secure_cookie("uname")
        if not user_id:
            return None
        if(self.isValidUser(user_id)):
            return user_id
        else: return None
    
    def get_template_path(self):
        return config.DEFAULT_FRONT_THEME
    
    def countPrivilege(self):
        '''根据预置的秘密计算一个时间相关的随机数，每分钟变一次，用来发布ROM的时候做验证。'''
        secret = config.AUTOPUB_SECRET
        salt = time.strftime("%Y-%m-%d %H:00",time.localtime(time.time()))
        ptoken = utils.sha256(secret+salt)
        print("hasPrivilege: ptoken is:",ptoken)
        return ptoken
        
    def hasPrivilege(self,ptoken):
        token = self.countPrivilege()
        print("hasPrivilege: token is:",token ," ptoken is:",ptoken)
        return token == ptoken

    def accessSelf(self,uname):
        '''判断参数用户名是否是自己'''
        curuname = self.current_user.decode()
        return curuname ==  uname

    def seeother(self,path):
        self.redirect(config.netpref['SCHEME']+"://"+config.netpref['SERVER_HOST']+":"+config.netpref['SERVER_PORT']+path)
    
    def unescape(self, the_str):
        return tornado.escape.xhtml_unescape(the_str)
    
    def mk2Html(self,markdown_text):
        html = utils.markdown2html(markdown_text)
        return html
        #return tornado.escape.xhtml_escape(html)
