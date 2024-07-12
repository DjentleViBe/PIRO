CC = /usr/bin/clang++

# Compiler flags
CFLAGS = -std=c++23 -Wall -g -Wno-deprecated -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64

MAKEFILE_DIR := $(CURDIR)
# Directories
SRCDIRS := $(MAKEFILE_DIR)/src $(MAKEFILE_DIR)/dependencies/backends
SRCDIR = src
BCKNDS = $(MAKEFILE_DIR)/dependencies/backends
INCDIR = $(MAKEFILE_DIR)/dependencies/include
OBJDIR = $(MAKEFILE_DIR)/bin/
BINDIR = $(MAKEFILE_DIR)/bin/
#LIBS = -lglfw.3.3 -lportaudio
LIBS = 
LIBDIR = $(MAKEFILE_DIR)/dependencies/library
SHAREDLIB = $(MAKEFILE_DIR)/sharedlib
FRAMEWORKS = -framework CoreFoundation -framework Accelerate -framework OpenCL
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
EXEC = $(BINDIR)/GIRO
OBJS := $(OBJS_C) $(OBJS_CPP)
# Default target
all: $(EXEC)

# Compile cpp source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS_C) $(HEADERS_CPP)| $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INCDIR)

# Link object files to create executable
$(EXEC): $(OBJS) | $(BINDIR)
	$(CC) $(CFLAGS) $(OBJS) -L$(LIBDIR) $(LIBS) $(FRAMEWORKS) -o $(EXEC)

# Create directories if they don't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

#$(BINDIR):
#	mkdir -p $(BINDIR)

setup:
	@echo "Running setup commands..."
#	cp prefs.json bin/.
	mkdir -p bin/assets
	cp -r dependencies/assets/* bin/assets/.

.PHONY: clean
# Clean up
clean:
	rm -rf $(OBJDIR) $(BINDIR)