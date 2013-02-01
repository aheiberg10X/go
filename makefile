CFLAGS = #--cubin #--ptxas-options=-v #-pg
INC = #-I/usr/local/cuda/include
LINKS = #-L/usr/local/cuda/lib64 -lcuda -lcudart
CMPLR = nvcc 

#all:  clean go
#all: cleango goonly
all: clean shared
#all: clean benchmark

benchmark: kernel.o 
	nvcc -o kernel kernel.o

shared: kernel.o godomain.o mcts.o go.o
	nvcc -o kernel godomain.o kernel.o mcts.o mcts_node.o go.o

kernel.o:
	nvcc -c kernel.cu

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

cleango:
	rm -f go.o go

clean:
	rm -rf *.o go kernel



#go: go.o gostate_struct.o godomain.o mcts.o 
	#${CMPLR} -o go ${LINKS} *.o 
