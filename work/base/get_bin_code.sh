#!/bin/bash

tools_dir="../../tools"
ida_path="${tools_dir}/idapro-9.0/ida64"
script1_path="${tools_dir}/getFuncAddr.py"
script2_path="${tools_dir}/getFuncCode.py"
export DISPLAY=:0
export QT_QPA_PLATFORM=offscreen
if [ "$#" -eq 0 ]; then
    $ida_path -A -S"${script2_path}" bin.exe
    rm bin.exe.id0 bin.exe.id1 bin.exe.id2 bin.exe.nam bin.exe.til func_addr.log

elif [ "$#" -eq 1 ]; then
    function_name="$1"
    $ida_path -A -S"${script1_path} ${function_name}" bin_symbol.exe
    rm bin_symbol.exe.id0 bin_symbol.exe.id1 bin_symbol.exe.id2 bin_symbol.exe.nam bin_symbol.exe.til
else
    exit 1
fi
