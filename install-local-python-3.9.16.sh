#!/bin/bash

cd ~
mkdir opt tmp

#Download packages
cd tmp
wget https://www.openssl.org/source/openssl-1.1.1t.tar.gz
tar -xzf openssl-1.1.1t.tar.gz
wget https://src.fedoraproject.org/lookaside/pkgs/bzip2/bzip2-1.0.6.tar.gz/00b516f4704d4a7cb50a1d97e6e8e15b/bzip2-1.0.6.tar.gz
tar -xvf bzip2-1.0.6.tar.gz
wget http://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz
tar -xvf Python-3.9.16.tgz
mv Python-3.9.16/ python-3.9.16/

export LD_LIBRARY_PATH=$HOME/opt/local/lib
export LD_RUN_PATH=$LD_LIBRARY_PATH
export LDFLAGS="-L$HOME/opt/local/lib"
export CPPFLAGS="-I$HOME/opt/local/include"
export CXXFLAGS=$CPPFLAGS
export CFLAGS=$CPPFLAGS

# Build openssl
cd openssl-1.1.1t
./config --prefix=$HOME/opt/local --openssldir=$HOME/opt/local/openssl shared
make
make install
cd ..

#Build bzip2
cd bzip2-1.0.6
make -f Makefile-libbz2_so
make clean
make
make PREFIX=$HOME/opt/local

cp -v bzip2-shared $HOME/opt/local/bin/bzip2
chmod a+x $HOME/opt/local/bin/bzip2
cp -f bzlib.h $HOME/opt/local/include
chmod a+r $HOME/opt/local/include/bzlib.h
cp -f libbz2.a $HOME/opt/local/lib
chmod a+r $HOME/opt/local/lib/libbz2.a

cp -av libbz2.so* $HOME/opt/local/lib
ln -sv libbz2.so.1.0 $HOME/opt/local/lib/libbz2.so
cd ..

#Build python
cd python-3.9.16
./configure --prefix=$HOME/opt/local --with-openssl=$HOME/opt/local
make
make install
cd ..
