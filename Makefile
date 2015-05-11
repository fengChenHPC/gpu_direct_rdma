nvcc=nvcc -O3
CFLAGS= -I/usr/local/cuda-7.0/include -L/usr/local/cuda-7.0/lib64
mpicc=mpic++ -O3

all: d2d oldway h2h allreduce

d2d: d2d.o lib.o
	$(mpicc) d2d.o lib.o -o $@ -lcudart $(CFLAGS)

d2d.o: d2d.cpp
	$(mpicc) -c $< $(CFLAGS)

allreduce: allreduce.o lib.o
	$(mpicc) allreduce.o lib.o -o $@ -lcudart $(CFLAGS)

allreduce.o: allreduce.cpp
	$(mpicc) -c $< $(CFLAGS)

oldway: oldway.o lib.o
	$(mpicc) oldway.o lib.o -o $@ -lcudart $(CFLAGS)

oldway.o: oldway.cpp
	$(mpicc) -c $< $(CFLAGS)

h2h: h2h.o lib.o
	$(mpicc) h2h.o lib.o -o $@ -lcudart $(CFLAGS)

h2h.o: h2h.cpp
	$(mpicc) -c $< $(CFLAGS)

lib.o: lib.cu	
	$(nvcc) -c lib.cu -arch=sm_35

clean:
	rm -f core *.o a.out d2d oldway h2h
