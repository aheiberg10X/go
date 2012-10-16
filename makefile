CFLAGS = #-pg

all: clean go

go: go.o gostate.o godomain.o mcts.o
	g++ go.o gostate.o godomain.o mcts.o mcts_node.o -o go ${CFLAGS}

mcts.o : mcts_node.o
	g++ -c mcts.cpp ${CFLAGS}

mcts_node.o: 
	g++ -c mcts_node.cpp ${CFLAGS}

godomain.o: stonestring.o
	g++ -c godomain.cpp ${CFLAGS}

stonestring.o:
	g++ -c stonestring.cpp ${CFLAGS}

go.o:
	g++ -c go.cpp ${CFLAGS}  

gostate.o: queue.o 
	g++ -c gostate.cpp ${CFLAGS}

queue.o:
	g++ -c queue.cpp ${CFLAGS}

clean:
	rm -rf *.o go 
