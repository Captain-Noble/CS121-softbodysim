# Windows + MinGW-w64 (PowerShell / cmd)
# Build with: mingw32-make
# Clean with: mingw32-make clean

CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS  := -lws2_32

TARGET   := PBDServer.exe
SRC      := PBDServer.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	-@del /Q $(TARGET) 2>NUL

.PHONY: all clean
