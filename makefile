CFLAGS= -pg -fopenmp
INC = #-I/usr/local/cuda/include
LINKS = #-L/usr/local/cuda/lib64 -lcuda -lcudart
CMPLR=g++ 

all: clean shared
#all : clean_omp omp
#all: clean_linktest linktest

######################################################################

omp:
	g++ omp.cpp -o omp ${CFLAGS}

#######################################################################

linktest: linktest.o
	g++ -o linktest ComputeValueBoard_binaryFeat linktest.o 

linktest.o :
	g++ -c linktest.cpp ${CFLAGS}

########################################################################

benchmark: kernel.o 
	${CMPLR} -o kernel kernel.o ${CFLAGS}

shared: kernel.o godomain.o mcts.o go.o
	${CMPLR} -o kernel godomain.o kernel.o mcts.o mcts_node.o go.o ${CFLAGS}

kernel.o:
	${CMPLR} -c kernel.cpp ${CFLAGS}

mcts.o : mcts_node.o
	${CMPLR} -c mcts.cpp ${CFLAGS}
	#g++ -c mcts.cpp ${CFLAGS}

mcts_node.o: 
	${CMPLR} -c mcts_node.cpp ${CFLAGS}

godomain.o: 
	${CMPLR} -c godomain.cpp ${INC} ${CFLAGS}

go.o:
	${CMPLR} -c go.cpp ${INC} ${CFLAGS} 

######################################################################

cleango:
	rm -f go.o go

clean_omp :
	rm -f omp omp.o

clean_linktest:
	rm -f linktest linktest.o

clean:
	rm -rf *.o go kernel



#go: go.o gostate_struct.o godomain.o mcts.o 
	#${CMPLR} -o go ${LINKS} *.o 
