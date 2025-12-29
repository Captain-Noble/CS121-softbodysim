# Windows + MinGW-w64 (PowerShell / cmd)
# Build with: mingw32-make
# Clean with: mingw32-make clean

CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude -pthread

# 关键：静态 libstdc++ / libgcc / winpthread，避免缺 DLL/入口点导致“闪退”
LDFLAGS  := -static -static-libgcc -static-libstdc++ -pthread -lws2_32

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

src/%.o: src/%.cpp include/PBDServer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-@del /Q $(TARGET) $(OBJ) 2>NUL

.PHONY: all clean
