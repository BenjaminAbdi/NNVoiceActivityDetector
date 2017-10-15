#!/bin/bash
search_dir="./"
cpt=0
folder="./output/"
base_name="fable"
for filename in `ls $search_dir`; do
		let cpt=cpt+1
		echo $cpt
    if [[ $filename == *".mp3"* ]]
		then
			sox $filename -c 1 $folder$base_name$cpt".wav"
		fi
done