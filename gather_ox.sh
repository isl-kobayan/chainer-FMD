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

function mkdirifnotexists() {
  if [ ! -e $1 ]; then
    mkdir $1
  fi
}

function gather_image_from_list() {
  if [ -e $2 ]; then
    for line in `cat $1`; do
      #echo "${line}"
      dir=${line%/*}
      mtl=${dir##*/}
      fname=${line##*/}
      mkdirifnotexists "$2/${mtl}"
      cp $line "$2/${mtl}/${fname}"
    done
  fi
}

function gather_image_from_list2() {
  OLDIFS=$IFS
  #ntl=( `cat $3` )
  if [ -e $2 ]; then
    IFS=$'\n'
    for line in `awk 'NR>1 {print}' $1`; do
      #echo "${line}"
      IFS=$'\t '
      arr=( `echo ${line}` )
      #echo ${arr[0]}
      #echo ${arr[1]}
      #echo ${arr[2]}

      filepath=${arr[0]}
      #pred=${ntl[${arr[2]}]}
      pred=${arr[2]}
      dir=${filepath%/*}
      mtl=${dir##*/}
      fname=${filepath##*/}
      mkdirifnotexists "${2}/${mtl}"
      cp $filepath "${2}/${mtl}/${pred}_${fname}"
    done
  fi
  IFS=$OLDIFS
}

filespath="./files.txt"
labelspath="./labels.txt"
imagelistpath="./image_list.txt"
num2labelpath="./num2label.txt"

#clearf $filespath
#clearf $labelspath
#clearf $imagelistpath
#clearf $num2labelpath



#in_dir="./image"
if [ $# -lt 1 ]; then
  echo "指定された引数は$#個です。" 1>&2
  echo "実行するには1個の引数が必要です。" 1>&2
  exit 1
fi

in_dir=${1%/*}
correct_dir="${in_dir}/correct"
incorrect_dir="${in_dir}/incorrect"
confusion_dir="${in_dir}/confusion"
correct_list="${in_dir}/correct.txt"
incorrect_list="${in_dir}/incorrect.txt"

clearf $correct_list
clearf $incorrect_list
clearf $correct_dir
clearf $incorrect_dir
clearf $confusion_dir

awk '{ if ( NR > 1 && $2 == $3 ) {print $1} }' $1 > $correct_list
awk '{ if ( NR > 1 && $2 != $3 ) {print $1} }' $1 > $incorrect_list

mkdir $correct_dir
mkdir $incorrect_dir
mkdir $confusion_dir

gather_image_from_list $correct_list $correct_dir
gather_image_from_list $incorrect_list $incorrect_dir
gather_image_from_list2 $1 $confusion_dir 
