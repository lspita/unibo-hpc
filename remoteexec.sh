#!/usr/bin/env bash

set -e  # exit on error

FILES=("$@")
MAIN_FILE="${FILES[0]}"
MAIN_BASENAME=$(basename "$MAIN_FILE")
MAIN_EXEC="${MAIN_BASENAME%.*}"
REMOTE_CC=${REMOTE_CC:-"gcc"}
REMOTE_CC_FLAGS=${REMOTE_CC_FLAGS:-""}

echo "Remote: $SSH_REMOTE"
echo "Main: $MAIN_FILE"
echo "Sources: ${FILES[@]}"

SOCKET_DIR=$(mktemp -d)
SOCKET="$SOCKET_DIR/socket"

cleanup() {
    if [ -S "$SOCKET" ]; then
        echo "Closing SSH connection"
        ssh -O exit -S "$SOCKET" "$SSH_REMOTE"
    else
        echo "Socket does not exist: $SOCKET" 1>&2
        exit 1
    fi
    rm -rf "$SOCKET_DIR"
}
trap cleanup EXIT
trap cleanup SIGTERM

echo "Connecting to ssh remote"
# -M: creates a "master" connection that other connections can piggyback on
# -N: no command
# -f: fork to background to proceed with script execution
# ControlPersist=yes: mantain connection valid even if something else is executed between ssh commands
ssh -M -S "$SOCKET" -Nf -o ControlPersist="yes" "$SSH_REMOTE"

echo "Copying files to remote"
for file in "${FILES[@]}"; do
    ssh -S "$SOCKET" "$SSH_REMOTE" "mkdir -p $(dirname $file)"
    scp -o ControlPath="$SOCKET" "$file" "$SSH_REMOTE:$file"
done

echo "Compiling files"
compile_command="$REMOTE_CC -o $MAIN_EXEC ${FILES[@]} $REMOTE_CC_FLAGS"
echo "$compile_command"
ssh -S "$SOCKET" "$SSH_REMOTE" "$compile_command"

exec_file="./$MAIN_EXEC"
echo "Running executable"
echo "$exec_file"
ssh -S "$SOCKET" "$SSH_REMOTE" "chmod +x $MAIN_EXEC && ./$MAIN_EXEC"
