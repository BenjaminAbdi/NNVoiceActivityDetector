#!/bin/bash
search_dir="./"
cpt=0
base_name="fable"
for filename in `ls $search_dir`; do
    if [[ $filename == *".wav"* ]]
		then
			filename_without_ext="${filename%-silenceless.*}"
            sox $filename -c 1 -r 16000 $filename_without_ext"-upgraded.wav"
            rm $filename
            rm $filename_without_ext".wav"
		fi
done
