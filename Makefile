OS := $(shell uname)
$(info Detected OS: $(OS))

ifeq ($(OS), Darwin)
    CC = /usr/bin/clang++
    MAKEFILE_DIR := $(CURDIR)
    $(info Current Directory: $(MAKEFILE_DIR))
	CFLAGS = -std=c++23 -Wall -g -Wno-deprecated -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
	LIBDIR = -L$(MAKEFILE_DIR)/dependencies/library/
	LIBS = 
	EXENAME = GIRO_devices
	FRAMEWORKS = -framework CoreFoundation -framework Accelerate -framework OpenCL
else
    CC = g++
    MAKEFILE_DIR=.
    $(info Current Directory: $(MAKEFILE_DIR))
	LIBDIR = -L$(MAKEFILE_DIR)/dependencies/library
	CFLAGS = -std=c++23 -Wno-deprecated
	LIBS = -lOpenCL
	EXENAME = GIRO_devices.exe
endif

# Directories
SRCDIRS := $(MAKEFILE_DIR)/src $(MAKEFILE_DIR)/dependencies/backends
SRCDIR = src
BCKNDS = $(MAKEFILE_DIR)/dependencies/backends
INCDIR = $(MAKEFILE_DIR)/dependencies/include
OBJDIR = $(MAKEFILE_DIR)/bin/
BINDIR = $(MAKEFILE_DIR)/bin/
#LIBS = -lglfw.3.3 -lportaudio

SHAREDLIB = $(MAKEFILE_DIR)/sharedlib
# Source files
SOURCES_C := $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.c))
SOURCES_CPP := $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.cpp))
# Object files
OBJS_C := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SOURCES_C))
OBJS_CPP := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES_CPP))
#header files 
HEADERS_C := $(wildcard $(INCDIR)/**/*.h)
HEADERS_CPP := $(wildcard $(INCDIR)/**/*.hpp)
# Executable
EXEC = $(BINDIR)$(EXENAME)
OBJS := $(OBJS_C) $(OBJS_CPP)
# Default target
all: $(EXEC)

# Compile cpp source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS_C) $(HEADERS_CPP)| $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Link object files to create executable
$(EXEC): $(OBJS) | $(BINDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $(LIBDIR) $(FRAMEWORKS) -o $(EXEC) $(OBJS) $(LIBS)

# Create directories if they don't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

#$(BINDIR):
#	mkdir -p $(BINDIR)

setup:
	@echo "Running setup commands..."
#	cp prefs.json bin/.
	mkdir -p $(MAKEFILE_DIR)/bin/assets
	cp -r $(MAKEFILE_DIR)/dependencies/assets/* $(MAKEFILE_DIR)/bin/assets/.

.PHONY: clean
# Clean up
clean:
	rm -rf $(OBJDIR) $(BINDIR)