#!/bin/bash
set -e
EXEC=exec

echo "Launched container with user: $USER, $(id -u):$(id -g)"
echo $@ | $EXEC $SHELL -li
