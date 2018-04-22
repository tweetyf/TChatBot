#!/usr/bin/env python3
#coding=utf-8
import hashlib,os

netpref ={}
#web 服务端API 和管理入口的地址 
netpref['SERVER_HOST']='127.0.0.1'
netpref['SERVER_PORT']='8080'
netpref['SCHEME']='http'
netpref['WSSCHEME']='ws'

#redis main DB:
netpref['REDIS_HOST']='127.0.0.1'
netpref['REDIS_PORT']=6379
netpref['REDIS_DB']=0
netpref['REDIS_PASSWORD']=None

DEFAULT_BOTNAME="蓝色章鱼哥"
DEFAULT_HEAD="/static/header.png"

#网站信息设置
DEFAULT_LOGO="/static/logo.png"
DEFAULT_SITENAME="TinfoHub"
DEFAULT_SITEINFO="Hmm.. interesting."

##给cookie设置，用于验证加密cookie，发布时必须修改！
COOKIE_SECRET="aaa"
##设置cookie的超时时间，默认半个小时。
COOKIE_EXPIRE=18000

##前台主题设置
DEFAULT_FRONT_THEME="templates/theme_default"

#ROOT path
ROOT_PATH=os.getcwd()+'/'
