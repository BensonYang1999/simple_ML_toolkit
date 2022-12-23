CXX = g++
CXXFLAGS = -O3 -std=c++17
MKLINCLUDE = /usr/include/mkl
MKLFLAGS = -lblas -llapacke -llapack -lm
PYINCLUDE = /usr/include/python3.8
PYFLAGS = -lpython3.8

main:
	$(CXX) $(CXXFLAGS) -I$(MKLINCLUDE) -I./ src/main.cpp -o a.out $(MKLFLAGS)
temp:
	$(CXX) $(CXXFLAGS) -I$(MKLINCLUDE) temp.cpp -o temp.out $(MKLFLAGS)
	./temp.out
py:
	$(CXX) $(CXXFLAGS) -shared -fPIC -I$(MKLINCLUDE) -I$(PYINCLUDE) -I./src `python3 -m pybind11 --includes` pytrans/py.cpp -o simpleml`python3-config --extension-suffix` $(PYFLAGS) $(MKLFLAGS)
