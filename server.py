#!/usr/bin/env python3
#coding=utf-8
# web.py In Memery of Aaron Swartz
# 2017.12.10: Switched into Tornado

import tornado.escape
from tornado import gen
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options

import config
import action
import action.admin

import os,sys

### Url mappings
handlers = [
    ##for web view
    ('/',                action.admin.Index),
    # Make url ending with or without '/' going to the same class
    ('/(.*)/',                   action.admin.redirect), 
]
class Application(tornado.web.Application):
    def __init__(self):
        settings = dict(
            #see: RequestHandler.get_template_path()
            #template_path=os.path.join(os.path.dirname(__file__), config.DEFAULT_FRONT_THEME),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
            cookie_secret=config.COOKIE_SECRET,
            login_url="/404",#no need to show any login url.
            default_handler_class=action.admin.ErrorPage,
            debug=False,
            static_hash_cache=False,
        )
        super(Application, self).__init__(handlers, **settings)

class EntryModule(tornado.web.UIModule):
    def render(self, entry):
        return self.render_string("modules/entry.html", entry=entry)
       
def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    if len(sys.argv)==2:
        http_server.listen(sys.argv[1])
    elif len(sys.argv)>=3:
        http_server.listen(sys.argv[2],address=sys.argv[1])
    else:
        print("To run this server: \npython server.py 8080 \npython server.py 127.0.0.1 8080")
        exit()
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
