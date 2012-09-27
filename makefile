all: clean go

go: go.o state.o
	g++ go.o state.o -o go

go.o:
	g++ -c go.cpp

state.o: 
	g++ -c state.cpp

clean:
	rm -rf *.o go
