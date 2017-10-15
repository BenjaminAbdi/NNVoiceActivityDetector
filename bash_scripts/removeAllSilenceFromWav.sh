#!/bin/bash
search_dir="./"
cpt=0
base_name="fable"
for filename in `ls $search_dir`; do
    if [[ $filename == *".wav"* ]]
		then
			filename_without_ext="${filename%.*}"
			sox $filename $filename_without_ext"-silenceless.wav" silence 1 0.1 1% -1 0.1 1%
		fi
done