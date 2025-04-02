from idaapi import *
from idc import *
from ida_auto import auto_wait
import ida_name
import idautils
import re

def readAddr():
    try:
        with open("func_addr.log",'r') as fp:
            addr = int(fp.read(), 16)
        return addr
    except:
        return 0

def writeCode(content):
    with open("func_code.log",'w') as f:
        f.write(content)

def main():
    addr = readAddr()
    if addr == 0:
        return
    try:
        cfunc = decompile(addr)
    except Exception as e:
        return
    writeCode(str(cfunc))

if __name__=="__main__":
    auto_wait()
    main()
    idaapi.qexit()

