#!/bin/bash

model_name=${1:-"help"}
usage="usage: ./sim_multi_clients.sh model_name [n_clients] [path/to/sim_test_ids.txt] [server_address] [server_port] [path/to/outputs]"

if [[ "$model_name" =~ help|--help|-h ]]
then
    echo $usage
    exit
else
    sim_test_ids_txt=${3:-"sim_test_ids.txt"}
    readarray -t test_ids < $sim_test_ids_txt
    n_clients=${2:-"${#test_ids[@]}"}
    server_address=${4:-"140.113.195.248"}
    server_port=${5:-"9999"}
    out=${6:-"out/CollabLoc"}

    outdir="$out/$model_name/"$n_clients"_clients"
    cmd=""
    mkdir -p $outdir
    for i in $( seq 1 ${n_clients} )
    do
        if [ $i -ne 1 ]
        then
            cmd+=" & "
        fi
        cmd+="python3 client_simulator.py --model_name $model_name --test_id ${test_ids[i-1]} --out $outdir --server_address $server_address --server_port $server_port"
    done
    eval $cmd
    echo "Done. You may find logs in $outdir"
fi