#!/usr/bin/env python3
#coding=utf-8

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

'''
dataset from https://github.com/fateleak/dgk_lost_conv
'''
def trainWithConv(chatbot, convFilename):
    print('Trainning with:',convFilename)
    chatbot.set_trainer(ListTrainer)
    xiaohuangjiC= open(convFilename,'r')
    txtCorpus=[]
    for line in xiaohuangjiC:
        if line.startswith('M '):
            line = line.replace('M ','').replace('/','').strip()
            txtCorpus.append(line)
            #print(line)
        elif line.startswith('E'):
            chatbot.train(txtCorpus)
            txtCorpus=[]

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
trainWithConv(chatbot, './static/fanzxl.conv')
trainWithConv(chatbot, './static/fk24.conv')
trainWithConv(chatbot, './static/haosys.conv')
trainWithConv(chatbot, './static/juemds.conv')
trainWithConv(chatbot, './static/laoyj.conv')
trainWithConv(chatbot, './static/lost.conv')
trainWithConv(chatbot, './static/prisonb.conv')
trainWithConv(chatbot, './static/xiaohuangji50w_nofenci.conv')
trainWithConv(chatbot, './static/dgk_shooter_z.conv')
## Add more training data here.

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
