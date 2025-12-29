# Windows + MinGW-w64 (PowerShell / cmd)
# Build with: mingw32-make
# Clean with: mingw32-make clean

CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS  := -lws2_32

TARGET   := PBDServer.exe

SRC := \
  src/main.cpp \
  src/Net.cpp \
  src/Server.cpp \
  src/Sim.cpp

OBJ := $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDFLAGS)

# compile each .cpp to .o next to it (src/*.o)
src/%.o: src/%.cpp include/PBDServer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-@del /Q $(TARGET) $(OBJ) 2>NUL

.PHONY: all clean
