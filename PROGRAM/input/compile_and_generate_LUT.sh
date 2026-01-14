#!/bin/bash

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./compile_and_generate_LUT.sh nbits num_k k_file"
    echo ""
    echo "Options:"
    echo "  -h, --help: Show this help"
  exit 0
fi

nbits=$1
NUMBETAS=$2
beta_file=$3

target=`echo $1 $2 | awk '{printf("create_LUT_nbits%02d_NB%02d",$1,$2)}'`

echo $target

gcc create_LUT.c -o $target -DNUMBITSPREBUSQUEDAS=${nbits} -DNUMBETAS=${NUMBETAS} -lm -lquadmath -Wall -Wshadow

./$target $beta_file
