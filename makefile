CFLAGS= -O3 -fopenmp
INC = #-I/usr/local/cuda/include
LINKS = #-L/usr/local/cuda/lib64 -lcuda -lcudart
CMPLR=/usr/bin/g++-4.4 

all: clean shared
#all : clean_omp omp
#all : clean_cross cross_corr

######################################################################

cross_corr:
	nvcc cross_corr_kernel.cu -o cross_kernel

######################################################################

omp:
	g++ omp.cpp -o omp ${CFLAGS}

########################################################################

benchmark: kernel.o 
	${CMPLR} -o kernel kernel.o ${CFLAGS}

shared: kernel.o mcts.o go.o
	${CMPLR} -o kernel queue.o bitmask.o gostate.o zobrist.o mcts.o mcts_node.o go.o ${CFLAGS} \
	-L/Volumes/export/isn/andrew/go/value_functions -lvalue2 \
	-L/usr/local/MATLAB/R2011b/bin/glnxa64 -lmx -leng -lmat -lut	

kernel.o:
	#${CMPLR} -c kernel.cpp ${CFLAGS}
	${CMPLR} -c queue.cpp bitmask.cpp gostate.cpp zobrist.cpp ${CFLAGS}

mcts.o : mcts_node.o
	${CMPLR} -c mcts.cpp ${CFLAGS} \
	-I/usr/local/MATLAB/R2011b/extern/include/ 

mcts_node.o: 
	${CMPLR} -c mcts_node.cpp ${CFLAGS}

#godomain.o: 
	#${CMPLR} -c godomain.cpp ${INC} ${CFLAGS}



go.o:
	${CMPLR} -c go.cpp ${INC} ${CFLAGS} \
	-I/usr/local/MATLAB/R2011b/extern/include/ 

######################################################################

cleango:
	rm -f go.o go

clean_cross:
	rm -f cross_kernel cross_corr_kernel.o

clean_omp :
	rm -f omp omp.o

clean_linktest:
	rm -f linktest linktest.o

clean_valuetest :
	rm -f valuetest valuetest.o

clean:
	rm -rf *.o go kernel

#old stuff when trying to get matlab code to play nice
#######################################################################

#valuetest: 
	#/usr/bin/g++-4.4 valuetest.cpp -o valuetest \
        #-I/usr/local/MATLAB/R2011b/extern/include/ \
        #-L/Volumes/export/isn/andrew/go/value_functions -lvalue2 \
        #-L/usr/local/MATLAB/R2011b/bin/glnxa64 -lmx -leng -lmat -lut	
#
#########################################################################
#
#linktest2:
	#/usr/bin/g++-4.4 linktest.cpp -o linktest \
		#-I/usr/local/MATLAB/R2011b/extern/include/ \
		#-L/Volumes/export/isn/andrew/go -lmyavg \
		#-L/usr/local/MATLAB/R2011b/bin/glnxa64 -lmx -leng -lmat -lut
#
#linktest: linktest.o
	#g++ -o linktest -L/usr/local/MATLAB/R2011b/bin/glnxa64 -leng -lmx -lmat -L/Volumes/export/isn/andrew/go/avg_c/distrib -lavg linktest.o
#
#linktest.o :
	#g++ -c linktest.cpp -I/Volumes/export/isn/andrew/go/avg_c/distrib -I/usr/local/MATLAB/R2011b/extern/include/ ${CFLAGS}
#
#
#
#go: go.o gostate_struct.o godomain.o mcts.o 
	#${CMPLR} -o go ${LINKS} *.o 
