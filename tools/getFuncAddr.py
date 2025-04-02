from idaapi import *
from idc import *
import ida_name
import idautils
import re

def writeAddr(addr):
    with open("func_addr.log",'w') as fp:            
        fp.write(f"0x{addr:X}")

def writelog(log):
    with open("a.log",'a') as f:
        f.write(log + "\n")

def main():
    if len(idc.ARGV) < 2:
        writelog("error")
        return
    input_funcname = idc.ARGV[1]
    pattern = r'(?<![a-zA-Z0-9])' + input_funcname + r'(?![a-zA-Z0-9])'
    regex = re.compile(pattern)
    
    for func in idautils.Functions():
        func_name = ida_name.get_name(func)
        if regex.search(func_name):
            writeAddr(func)
            break

def PLUGIN_ENTRY():
    auto_wait()
    main()
    idaapi.qexit()

if __name__ == "__main__":
    PLUGIN_ENTRY()