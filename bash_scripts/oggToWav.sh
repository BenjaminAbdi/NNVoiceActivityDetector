search_dir="./"
folder="./output/"
base_name="fable"
cpt=0
for filename in `ls $search_dir`; do
	filename_without_ext="${filename%.*}"
	let cpt=cpt+1
  if [[ $filename == *".ogg"* ]]
	then
		#echo $filename "sound-"$cpt".wav"
		sox $filename -c 1 "sound-"$cpt".wav"
	fi
done
