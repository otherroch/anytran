#!/bin/bash

set -e

echo "Starting container setup..."

source venv/bin/activate
anytran $*

exec "$@"
~
