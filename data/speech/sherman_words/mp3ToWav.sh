#!/bin/bash
search_dir="./"
cpt=0
folder="./output/"
base_name="sherman"
for filename in `ls $search_dir`; do
		let cpt=cpt+1
		echo $cpt
    if [[ $filename == *".mp3"* ]]
		then
			sox $filename -c 1 $base_name$cpt".wav"
		fi
done
