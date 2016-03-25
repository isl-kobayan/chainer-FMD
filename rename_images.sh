#!/bin/bash

# delete file if exists
function clearf() {
  if [ -e $1 ]; then
    if [ -f $1 ] ; then
      rm $1
    elif [ -d $1 ] ; then
      rm -r $1
    fi
  fi
}


filespath="./files.txt"
labelspath="./labels.txt"
imagelistpath="./image_list.txt"
num2labelpath="./num2label.txt"

clearf $filespath
clearf $labelspath
clearf $imagelistpath
clearf $num2labelpath


in_dir="./image"
if [ $# -ge 1 ]; then
  in_dir=$1
fi

out_dir="${in_dir}_rename"
clearf $out_dir
mkdir $out_dir
files="${in_dir}/*"
dirary=()

for filepath in $files; do
  if [ -d $filepath ] ; then
    dirary+=("$filepath")
  fi
done

labelnum=0
#echo "ディレクトリ一覧"
for d in ${dirary[@]}; do
  #echo ${d##*/} 1>> $num2labelpath
  m_files="${d}/*"
  echo -e "${out_dir}/${d##*/}"
  mkdir "${out_dir}/${d##*/}"
  fcount=0
  for m_file in $m_files; do
    if [ -f "$m_file" ] ; then
      if [[ "$m_file" =~ ^.*\.(jpg|png|bmp)$ ]]; then
        #fileary+=("$filepath")
        #echo $m_file 1>> $filespath
        #echo $labelnum 1>> $labelspath
	#echo -e "${m_file}\t${labelnum}" 1>> $imagelistpath
	cp "${m_file}" "${out_dir}/${d##*/}/${fcount}.${m_file##*.}"
	let fcount=${fcount}+1
      fi
    fi
  done
  let labelnum=${labelnum}+1
done
