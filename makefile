PROG := effnetonc
SRCS := $(wildcard *.cpp) 
OBJS := $(SRCS:%.cpp=%.o)
DEPS := $(SRCS:%.cpp=%.d)

CC := g++
CCFLAGS := 
INCLUDEPATH := 
LIBPATH := 
LIBS := 


all: $(DEPENDS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LIBPATH) $(LIBS)

.cpp.o:
	$(CC) $(CCFLAGS) $(INCLUDEPATH) -DMTR_ENABLED -MMD -MP -MF $(<:%.cpp=%.d) -c $< -o $(<:%.cpp=%.o)


.PHONY: clean
clean:
	$(RM) $(PROG) $(OBJS) $(DEPS)

-include $(DEPS)