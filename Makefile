# ─────────────────────────────────────────────────────────────────────
#  turkish-tokenizer Makefile
#
#  Targets:
#    make              — build library + CLI tools
#    make lib          — build static library only
#    make tools        — build CLI tools (links against lib)
#    make tests        — build and run tests
#    make clean        — remove all build artifacts
# ─────────────────────────────────────────────────────────────────────

CC       = gcc
CFLAGS   = -Wall -Wextra -Wpedantic -std=c11 -O2 -Iinclude
LDFLAGS  =
AR       = ar
ARFLAGS  = rcs

# Debug build: make DEBUG=1
ifdef DEBUG
  CFLAGS += -g -O0 -DDEBUG -fsanitize=address,undefined
  LDFLAGS += -fsanitize=address,undefined
endif

# ── Directories ──────────────────────────────────────────────────────

SRC_DIR   = src
INC_DIR   = include
TOOL_DIR  = tools
TEST_DIR  = tests
BUILD_DIR = build

# ── Library sources ──────────────────────────────────────────────────

LIB_SRCS  = $(SRC_DIR)/unicode.c \
            $(SRC_DIR)/vocab.c   \
            $(SRC_DIR)/io.c      \
            $(SRC_DIR)/bpe.c     \
            $(SRC_DIR)/tokenizer.c

LIB_OBJS  = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(LIB_SRCS))
LIB_NAME  = $(BUILD_DIR)/libtokenizer.a

# ── CLI tools ────────────────────────────────────────────────────────

TOOL_SRCS = $(TOOL_DIR)/train.c   \
            $(TOOL_DIR)/encode.c  \
            $(TOOL_DIR)/decode.c  \
            $(TOOL_DIR)/inspect.c

TOOL_BINS = $(patsubst $(TOOL_DIR)/%.c, $(TOOL_DIR)/%, $(TOOL_SRCS))

# ── Tests ────────────────────────────────────────────────────────────

TEST_SRCS = $(TEST_DIR)/test_unicode.c   \
            $(TEST_DIR)/test_bpe.c       \
            $(TEST_DIR)/test_roundtrip.c \
            $(TEST_DIR)/test_turkish.c

TEST_BINS = $(patsubst $(TEST_DIR)/%.c, $(BUILD_DIR)/%, $(TEST_SRCS))

# ── Phony targets ───────────────────────────────────────────────────

.PHONY: all lib tools tests clean run-tests

all: lib tools

lib: $(LIB_NAME)

tools: $(TOOL_BINS)

tests: $(TEST_BINS) run-tests

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TOOL_BINS)

# ── Build rules ──────────────────────────────────────────────────────

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile library sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Archive into static library
$(LIB_NAME): $(LIB_OBJS)
	$(AR) $(ARFLAGS) $@ $^

# Link CLI tools
$(TOOL_DIR)/train: $(TOOL_DIR)/train.c $(LIB_NAME)
	$(CC) $(CFLAGS) $< -o $@ -L$(BUILD_DIR) -ltokenizer $(LDFLAGS)

$(TOOL_DIR)/encode: $(TOOL_DIR)/encode.c $(LIB_NAME)
	$(CC) $(CFLAGS) $< -o $@ -L$(BUILD_DIR) -ltokenizer $(LDFLAGS)

$(TOOL_DIR)/decode: $(TOOL_DIR)/decode.c $(LIB_NAME)
	$(CC) $(CFLAGS) $< -o $@ -L$(BUILD_DIR) -ltokenizer $(LDFLAGS)

$(TOOL_DIR)/inspect: $(TOOL_DIR)/inspect.c $(LIB_NAME)
	$(CC) $(CFLAGS) $< -o $@ -L$(BUILD_DIR) -ltokenizer $(LDFLAGS)

# Compile and link tests
$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.c $(LIB_NAME) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@ -L$(BUILD_DIR) -ltokenizer $(LDFLAGS)

# Run all tests
run-tests: $(TEST_BINS)
	@echo "═══════════════════════════════════════"
	@echo "  Running tests..."
	@echo "═══════════════════════════════════════"
	@failed=0; \
	for t in $(TEST_BINS); do \
		echo "\n── $$t ──"; \
		if $$t; then \
			echo "  PASS"; \
		else \
			echo "  FAIL"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "\n═══════════════════════════════════════"; \
	if [ $$failed -eq 0 ]; then \
		echo "  All tests passed!"; \
	else \
		echo "  $$failed test(s) failed."; \
		exit 1; \
	fi