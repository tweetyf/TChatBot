#!/usr/bin/env python3
#coding=utf-8

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

'''
chatbot = ChatBot("myBot")
# 使用英文语料库训练它
# 开始对话 
chatbot.get_response("Hello, how are you today?")
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
deepThought = ChatBot("deepThought")
deepThought.set_trainer(ChatterBotCorpusTrainer)
# 使用中文语料库训练它
deepThought.train("chatterbot.corpus.chinese")  # 语料库
'''
# Create a new instance of a ChatBot
chatbot = ChatBot(
    "Terminal",
    #storage_adapter="chatterbot.storage.SQLStorageAdapter",
    #database="./chatterbot.database.db",
    storage_adapter='chatterbot.storage.MongoDatabaseAdapter',
    database='chatterbot-database',
    logic_adapters=[
        "chatterbot.logic.BestMatch",
    #    "chatterbot.logic.MathematicalEvaluation",
    #    "chatterbot.logic.TimeLogicAdapter",
    ],
    input_adapter="chatterbot.input.TerminalAdapter",
    output_adapter="chatterbot.output.TerminalAdapter",
    read_only=False,
    
)
chatbot.set_trainer(ChatterBotCorpusTrainer)
chatbot.train("chatterbot.corpus.english")
chatbot.train("chatterbot.corpus.chinese")
xiaohuangjiC= open('./static/xiaohuangji50w_nofenci.conv','r')
txtCorpus=[]
for line in xiaohuangjiC:
    if line.startswith('M '):
        line = line.replace('M ','').strip()
        txtCorpus.append(line)
        #print(line)
    else: continue
chatbot.set_trainer(ListTrainer)
chatbot.train(txtCorpus)

print("Type something to begin...")

# The following loop will execute each time the user enters input
while True:
    try:
        # We pass None to this method because the parameter
        # is not used by the TerminalAdapter
        bot_input = chatbot.get_response("")
        #print(bot_input)
    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break
