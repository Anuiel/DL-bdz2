#!/bin/sh
GIT_HASH=$(git log --pretty=format:'%H' -n 1)
env/bin/python3 src/main.py --git-hash $GIT_HASH