all: clean go

go: go.o gostate.o godomain.o mcts.o
	g++ go.o gostate.o godomain.o mcts.o -o go

mcts.o : mcts_node.o
	g++ -c mcts.cpp

mcts_node.o: 
	g++ -c mcts_node.cpp

godomain.o: stonestring.o
	g++ -c godomain.cpp

stonestring.o:
	g++ -c stonestring.cpp

go.o:
	g++ -c go.cpp

gostate.o: 
	g++ -c gostate.cpp

clean:
	rm -rf *.o go
