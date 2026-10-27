#!/bin/bash
set -eu
cd "$(dirname "$0")"
python3 -c 'from bot import load_policy; load_policy(); print("Validated trained policy.pt")'
