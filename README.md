# TChatBot
a web based chat bot. with [chatterbot](https://chatterbot.readthedocs.io/en/stable/index.html) and [tornado](https://www.tornadoweb.org)

## requriment
mongodb:
```
sudo apt install mongodb python3-pip
```

python3
```
pip3 install jieba chatterbot tornado markdown
```

## usage:

- If it is your first time run this , you need to train bot.
```
python3 TrainChatterBot.py
```
- Run the server:
```
python3 server 8080
```

## websocket Interface:
When this web server started, there will be a websocket interface started as well.

websocket url:
```
 ws://localhost:8080/wschat
```

On websocket connected, client will recive a message from server wtich is in JSON format:
```
 {
 'id':id,  // server's id.
 'm':msg   // server's greeting message.
 }
```

Send your chat message to server:
```
 {
 'id':id,  // client's conversation id.
 'm':msg   // client's message.
 }
```

Then get response from server:
```
 {
 'id':id,  // client's conversation id.
 'm':msg   // server's message.
 }
```
