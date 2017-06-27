# Makefile for md5

TARGET=md5_par
OBJS=md5_par.o md5_bmark_par.o
CFLAGS=-O2 -g -D_GNU_SOURCE -std=c99
LDFLAGS= -fopenmp
CC=gcc

all: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET) -fopenmp

%.o : %.c
	$(CC) $(CFLAGS) -c $< -fopenmp

clean:
	rm -f *.o $(TARGET) output.txt
