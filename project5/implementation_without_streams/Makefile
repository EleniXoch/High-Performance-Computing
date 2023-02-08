CC = nvcc
CFLAGS = -g
RM = /bin/rm -f
all: main

main: main.o contrast-enhancement.o histogram-equalization.o
	$(CC) $(CFLAGS) -o main main.o contrast-enhancement.o histogram-equalization.o

main.o: main.cu 
	$(CC) $(CFLAGS) -c main.cu

contrast-enhancement.o: contrast-enhancement.cu histogram-equalization.o
	$(CC) $(CFLAGS) -c contrast-enhancement.cu histogram-equalization.o
	
histogram-equalization.o: histogram-equalization.cu 
	$(CC) $(CFLAGS) -c histogram-equalization.cu 
clean:
	$(RM) *.o main histogram-equalization contrast-enhancement
