BIN_DIR := @CMAKE_BINARY_DIR@
PRESET := @CMAKE_PRESET@
TARGETS_DIR := @TARGETS_DIR_ABS@
CMAKE := cmake
CTEST := ctest
PRESET_FLAG := --preset @CMAKE_PRESET@

TARGET ?= @EXEC_SOURCE_NAME@

CMAKE_SILENT_FLAG := -s

# Capture additional arguments
RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(RUN_ARGS):;@:)

default_target: build

build: $(BIN_DIR)
	$(CMAKE) --build $(PRESET_FLAG) -- $(CMAKE_SILENT_FLAG)

clean:
	-$(CMAKE) -E rm -rf $(BIN_DIR)/*
	-$(CMAKE) -E rm -rf $(BIN_DIR)/.*

reset: clean
	$(CMAKE) $(PRESET_FLAG)

test: build
	$(CTEST) $(PRESET_FLAG) $(RUN_ARGS)

run: build
ifneq ($(RUN_ARGS),)
	$(TARGETS_DIR)/$(RUN_ARGS)
else
	$(TARGETS_DIR)/$(TARGET)
endif
