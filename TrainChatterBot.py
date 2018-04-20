#!/usr/bin/env python3
#coding=utf-8

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

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
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    logic_adapters=[
        "chatterbot.logic.BestMatch",
    #    "chatterbot.logic.MathematicalEvaluation",
    #    "chatterbot.logic.TimeLogicAdapter",
    ],
    input_adapter="chatterbot.input.TerminalAdapter",
    output_adapter="chatterbot.output.TerminalAdapter",
    database="./chatterbot.database.db"
)
chatbot.set_trainer(ChatterBotCorpusTrainer)
chatbot.train("chatterbot.corpus.english")
chatbot.train("chatterbot.corpus.chinese")
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
