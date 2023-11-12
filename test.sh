#!/usr/bin/bash
if ! pgrep mongo; then
    echo "!! Be sure to start mongodb first !!"
fi

PYTHONPATH=./src ./.venv/bin/python -m pytest test/
