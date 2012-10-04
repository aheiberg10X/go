all: clean go

go: go.o gostate.o godomain.o
	g++ go.o gostate.o godomain.o -o go

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
