CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g 
LDFLAGS=-g 
LDLIBS=-lpthread -lX11 -lboost_program_options

all: render abstract_render

render: render.cpp
	$(CXX) $(LDFLAGS) render.cpp -o render $(LDLIBS) 

abstract_render: abstract_render.cpp
	$(CXX) $(LDFLAGS) abstract_render.cpp -o abstract_render $(LDLIBS) 

clean:
	$(RM) $(OBJS) render abstract_render
