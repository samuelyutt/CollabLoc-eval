#!/bin/bash

model_name=${1:-"help"}
usage="usage: ./sim_multi_clients.sh model_name [path/to/sim_test_ids.txt] [n_clients]"

if [[ "$model_name" =~ help|--help|-h ]]
then
    echo $usage
    exit
else
    sim_test_ids_txt=${2:-"sim_test_ids.txt"}
    readarray -t test_ids < $sim_test_ids_txt
    n_clients=${3:-"${#test_ids[@]}"}
    out=${4:-"out/CollabLoc"}

    outdir="$out/$model_name/"$n_clients"_clients"
    cmd=""
    mkdir -p $outdir
    for i in $( seq 1 ${n_clients} )
    do
        if [ $i -ne 1 ]
        then
            cmd+=" & "
        fi
        cmd+="python3 client_simulator.py --model_name $model_name --test_id ${test_ids[i-1]} --out $outdir"
    done

    eval $cmd
fi