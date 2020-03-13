CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-O3 -std=c++17
LDLIBS=-lpthread -lX11 -lboost_program_options

all: render abstract_render_zono abstract_render_bbb

render: render.cpp
	$(CXX) $(CPPFLAGS) render.cpp -o render $(LDLIBS) 

abstract_render_zono: abstract_render.cpp
	$(CXX) $(CPPFLAGS) abstract_render.cpp -o abstract_render_zono $(LDLIBS) -DABSTRACT_ZONO 

abstract_render_bbb: abstract_render.cpp
	$(CXX) $(CPPFLAGS) abstract_render.cpp -o abstract_render_bbb $(LDLIBS) -DABSTRACT_BBB 

clean:
	$(RM) $(OBJS) render abstract_render_zono abstract_render_bbb
