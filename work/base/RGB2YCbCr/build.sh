#!/bin/bash


clang-18 -target i686-w64-mingw32 -c bin.ll -o bin.o
i686-w64-mingw32-gcc -Wl,--gc-sections -g -o bin_symbol.exe bin.o
cp bin_symbol.exe bin.exe
i686-w64-mingw32-strip bin.exe