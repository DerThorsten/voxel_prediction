from colorama import Fore, Back, Style
import os
from progressbar import *         

def reraise(future):
    ex = future.exception()
    if ex :
        raise ex

def redStr(s):
    rs = Fore.RED+s+Fore.RESET + Back.RESET + Style.RESET_ALL
    return rs

def greenStr(s):
    rs = Fore.GREEN+s+Fore.RESET + Back.RESET + Style.RESET_ALL
    return rs

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def getPbar(maxval,name=""):
    cname = redStr(name)
    widgets = [' %s: '%cname, Percentage(), ' ', Bar(marker='*',left='[',right=']'),
           ' ',ETA()] #see docs for other options

    pbar = ProgressBar(widgets=widgets, maxval=maxval)
    #pbar.start()
    return pbar

def rmLastAxisSingleton(shape):
    if shape[-1] == 1:
        return shape[0:-1]
    else:
        return shape


def appendSingleton(slices):
    return slices +[ slice(0,1)]
