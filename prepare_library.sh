#!/bin/bash

# called by cmake to prepare a library by running a instrumentation pass on all object files passed as arguments

LOG=/tmp/liblog.txt
TMPDIR=/tmp/original_mods

[ -d $TMPDIR ] && rm $TMPDIR/*
[ ! -d $TMPDIR ] && mkdir $TMPDIR
[ -f $LOG ] && rm $LOG

debug=

if [ $# -eq 1 ]; then
  echo "Pass original bc files as arguments!"
  exit 1
fi

IS_APPROX=$1
PASS_PLUGIN=$2

for file in "${@:3}"; do
  echo "Applying for file $file" | tee -a $LOG
  DIR=$(dirname $file) 
  BASE_FILE=$(basename $file)

  BC_FILE=${file%.o}.bc

  $debug cd $DIR
  #if [ "$IS_APPROX" == "true" ]
  #then
  #  PREFIX=app
  #else
  #  PREFIX=prec
  #fi
  PREFIX=base

  if [ ! -f "$PREFIX"_$BASE_FILE ]
  then
    $debug cp $file "$PREFIX"_$BASE_FILE
  fi
  $debug opt -load-pass-plugin $PASS_PLUGIN --passes="annotation2metadata,forceattrs,inferattrs,coro-early,function<eager-inv>(lower-expect,sroa<modify-cfg>,early-cse<>,callsite-splitting),openmp-opt,ipsccp,called-value-propagation,globalopt,function(mem2reg),require<globals-aa>,function(invalidate<aa>),require<profile-summary>,cgscc(devirt<4>(inline<only-mandatory>,inline,function-attrs,argpromotion,openmp-opt-cgscc,function<eager-inv>(sroa<modify-cfg>,early-cse<memssa>,speculative-execution,correlated-propagation,libcalls-shrinkwrap,tailcallelim,reassociate,require<opt-remark-emit>,sroa<modify-cfg>,vector-combine,mldst-motion<no-split-footer-bb>,sccp,bdce,correlated-propagation,adce,memcpyopt,dse,coro-elide),coro-split)),deadargelim,coro-cleanup,globalopt,globaldce,elim-avail-extern,rpo-function-attrs,recompute-globalsaa,function<eager-inv>(float2int,lower-constant-intrinsics,loop-distribute,inject-tli-mappings,slp-vectorizer,vector-combine,transform-warning,sroa<preserve-cfg>,require<opt-remark-emit>,alignment-from-assumptions,loop-sink,instsimplify,div-rem-pairs,tailcallelim),globaldce,constmerge,cg-profile,rel-lookup-table-converter,function(annotation-remarks),instrappfn<$IS_APPROX>,verify" $file -o $file
 
  status=$?
  cp $file $TMPDIR
  # revert copy if failed to apply pass
  if test $status -ne 0 
  then
    echo "trouble applying passes to file $file" | tee -a $LOG 1>&2
    mv "$PREFIX"_$BASE_FILE $file
  fi

  $debug llc --filetype=obj --relocation-model=pic $file -o $file 

done
