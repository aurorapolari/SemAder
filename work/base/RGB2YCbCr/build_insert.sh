#!/bin/bash


clang-18 -target i686-w64-mingw32 -c bin_insert.ll -o bin_insert.o
i686-w64-mingw32-gcc -Wl,--gc-sections -g -o bin_insert_symbol.exe bin_insert.o
cp bin_insert_symbol.exe bin_insert.exe
i686-w64-mingw32-strip bin_insert.exe