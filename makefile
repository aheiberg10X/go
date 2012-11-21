CFLAGS = #-pg
INC = #-I/usr/local/cuda/include
LINKS = #-L/usr/local/cuda/lib64 -lcuda -lcudart
CMPLR = nvcc 

#all:  clean go
#all: cleango goonly
all: clean shared

shared: kernel.o godomain.o
	nvcc -o kernel godomain.o kernel.o

kernel.o:
	nvcc -c kernel.cu

goonly: go.o
	${CMPLR} -o go ${LINKS} *.o 

go: go.o gostate_struct.o godomain.o mcts.o 
	${CMPLR} -o go ${LINKS} *.o 

mcts.o : mcts_node.o
	${CMPLR} -c mcts.cpp ${CFLAGS}

mcts_node.o: 
	${CMPLR} -c mcts_node.cpp ${CFLAGS}

godomain.o: 
	${CMPLR} -c godomain.cpp ${INC} ${CFLAGS}

#stonestring.o:
	#g++ -c stonestring.cpp ${CFLAGS}

go.o:
	${CMPLR} -c go.cpp ${INC} ${CFLAGS}  

#had to place implementations in shared_mem.cu
#more generally, code needs to be in same file as kernel
#gostate_struct.o: queue.o bitmask.o
	#${CMPLR} -c gostate_struct.cu ${CFLAGS} 

#queue.o:
	#${CMPLR} -c queue.cu ${CFLAGS}

#bitmask.o:
	#${CMPLR} -c bitmask.cu ${CFLAGS}

cleango:
	rm -f go.o go

clean:
	rm -rf *.o go kernel
