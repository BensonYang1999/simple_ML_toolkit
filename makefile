SHELL := /bin/bash

CXX = g++
CXXFLAGS = -O3 -std=c++17

MKLINCLUDE = /usr/include/mkl
MKLFLAGS = -lblas -llapacke -llapack -lm

IMKLINCLUDE = /opt/intel/oneapi/mkl/latest/include
IMKLFLAGS = -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_ilp64 -lm -ldl

PYINCLUDE = /usr/include/python3.8
PYFLAGS = -lpython3.8

main:
	$(CXX) $(CXXFLAGS) -I$(MKLINCLUDE) -I./ src/main.cpp -o a.out $(MKLFLAGS)
py:
	$(CXX) $(CXXFLAGS) -shared -fPIC -I$(MKLINCLUDE) -I$(PYINCLUDE) -I./src `python3 -m pybind11 --includes` pytrans/py.cpp -o simpleml`python3-config --extension-suffix` $(PYFLAGS) $(MKLFLAGS)
	cp *.so tests/

intel:
	source /opt/intel/oneapi/setvars.sh intel64 --force && $(CXX) $(CXXFLAGS) -I$(IMKLINCLUDE) -I./ src/main.cpp -o a.out $(IMKLFLAGS)
# $(CXX) $(CXXFLAGS) -I$(IMKLINCLUDE) -I./ src/main.cpp -o a.out $(IMKLFLAGS)
intelpy:
	source /opt/intel/oneapi/setvars.sh intel64 --force && $(CXX) $(CXXFLAGS) -shared -fPIC -I$(IMKLINCLUDE) -I$(PYINCLUDE) -I./src `python3 -m pybind11 --includes` pytrans/py.cpp -o simpleml`python3-config --extension-suffix` $(PYFLAGS) $(IMKLFLAGS)
	cp *.so tests/

test: intelpy
	source /opt/intel/oneapi/setvars.sh intel64 --force && cd tests && python3 test.py
