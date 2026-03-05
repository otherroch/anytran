#!/bin/bash

set -e

echo "Starting container setup..."

source venv/bin/activate

#tail -f /dev/null
anytran $*


exec "$@"
~
