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
  if [ -e $2 ]; then
    IFS=$'\n'
    l=0
    for line in `cat $1`; do
      #echo "${line}"
      IFS=$'\t '
      c=0
      for col in `echo ${line}`; do
        #echo ${col}
        FORMAT=$(printf "%04d_%02d" ${l} ${c})
        convert -gravity center -crop 224x224+0+0 ${col} "${2}/${FORMAT}.jpg"
        #echo ${c}
        c=$((c+1))
      done
      FORMAT=$(printf "${2}/%04d_*" ${l})
      convert +append ${FORMAT} "${3}/${l}.jpg"
      l=$((l+1))
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
acts_dir="${in_dir}/acts"
acts2_dir="${in_dir}/acts2"

clearf $acts_dir
clearf $acts2_dir

mkdir $acts_dir
mkdir $acts2_dir

#usage: gather_acts.sh ***/actsargs.csv
gather_image_from_list2 $1 $acts_dir $acts2_dir
