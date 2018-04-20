#!/usr/bin/env python3
#coding=utf-8
import hashlib,os

netpref ={}
#web 服务端API 和管理入口的地址 
netpref['SERVER_HOST']='127.0.0.1'
netpref['SERVER_PORT']='8080'
netpref['SCHEME']='http'

#redis main DB: articals, settigns, users
netpref['REDIS_HOST']='127.0.0.1'
netpref['REDIS_PORT']=6379
netpref['REDIS_DB']=0
netpref['REDIS_PASSWORD']=None

DEFAULT_HEAD="/static/images/default_head.png"
DEFAULT_CAPTION="/static/images/default_caption.png"
DEFAULT_COVER="/static/images/default_cover.png"
DEFAULT_APPCAPTION="/static/images/appdefault.png"
DEFAULT_APPCOVER="/static/images/appdefault_cover.png"

#网站信息设置
DEFAULT_LOGO="/static/images/default_logo_philosoraptor.png"
DEFAULT_SITENAME="TinfoHub"
DEFAULT_SITEINFO="Hmm.. interesting."

##给cookie设置，用于验证加密cookie，发布时必须修改！
COOKIE_SECRET="bc4c516d9ab23c22c65ea2483aae5ba34c907f46f6d8dae11ece24004f486330"
##设置cookie的超时时间，默认半个小时。
COOKIE_EXPIRE=18000

##前台主题设置
DEFAULT_FRONT_THEME="templates/theme_default"

#ROOT path
ROOT_PATH=os.getcwd()+'/'