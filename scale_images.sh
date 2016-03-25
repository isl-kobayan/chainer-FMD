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

img_l="256"
if [ $# -ge 1 ]; then
  img_l=$1
fi

in_dir="./image"
out_dir=$in_dir$img_l
files="${in_dir}/*"
dirary=()

#rm -r $out_dir
clearf $out_dir
mkdir $out_dir

for filepath in $files; do
  if [ -d $filepath ] ; then
    dirary+=("$filepath")
  fi
done

labelnum=0
#echo "ディレクトリ一覧"
for d in ${dirary[@]}; do
  mkdir "${out_dir}/${d##*/}"
  m_files="${d}/*"
  for m_file in $m_files; do
    if [ -f $m_file ] ; then
      if [[ "$m_file" =~ ^.*\.(jpg|png|bmp)$ ]]; then
        #scale image
        convert -crop 384x384+64+0 -geometry "${img_l}x${img_l}" $m_file "${out_dir}/${d##*/}/${m_file##*/}"
	#echo "$img_l"
      fi
    fi
  done
  let labelnum=${labelnum}+1
done
