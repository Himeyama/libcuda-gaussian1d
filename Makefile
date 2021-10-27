NVCC = nvcc
 CXX = /usr/bin/g++
  CC = /usr/bin/gcc
 OPT = -ccbin $(CC) -std=c++11 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75

libgaussian1d.so: gaussian.cu
	$(NVCC) $(OPT) --shared -Xcompiler -fPIC $^ -o $@

test: test.cpp
	$(NVCC) $(OPT) -lgaussian1d -L. -I. $^ -o $@

test1: test.cu
	$(NVCC) $(OPT) $^ -o $@

install: libgaussian1d.so
	install -s $^ $(libdir)
	cp gaussian1d.hpp $(incdir)