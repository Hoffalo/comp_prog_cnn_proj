CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -g
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
BIN = bin
TARGET = $(BIN)/train

all: $(TARGET)

$(TARGET): $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lm

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(OBJ) $(BIN)

.PHONY: all clean
