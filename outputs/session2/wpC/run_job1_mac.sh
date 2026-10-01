#!/bin/bash
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
printf '%s\n' pair_d16_p000 pair_d16_p180 pair_d16_p090 pair_d16_p045 pair_d16_p135 twin_d16_p180 | xargs -P 2 -n 1 outputs/session2/wpC/run_one_mac.sh
