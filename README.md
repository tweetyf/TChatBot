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

## Deployment
You might want deploy the service behind nginx proxy server. To do this, you need set proxy headers for websocket.
```
server {
	listen 80 default_server;
	listen [::]:80 default_server ipv6only=on;
  root /path/to/your/server;

	server_name localhost;
	charset utf-8;
	access_log  /var/log/nginx/nginx.access.log;
	location / {
	    proxy_pass http://127.0.0.1:8080/;
            proxy_set_header Host $host;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 1d;
	}
    location /static {
        root /path/to/your/server;
        autoindex on;
        autoindex_exact_size off;
        autoindex_localtime on;
        charset utf-8;
        if (-f $request_filename) {
          rewrite ^/static/(.*)$  /static/$1 break;
        }
    }
}

```
