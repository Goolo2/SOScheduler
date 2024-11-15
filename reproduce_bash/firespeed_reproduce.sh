strategy='CEPY' 


if [ "$strategy" = 'CEPY' ]; then
    folder='confentropy'
fi


main='main.py'


for update_interval in $(seq 3 1 3); do
    echo $update_interval
    # for seed in $(seq 15 1 60); do
    for seed in $(seq 15 1 15); do
        echo $seed
        python $main --config "./simulations/configs/FSPEED_${strategy}.yaml" --update_interval $update_interval --seed $seed --save_dir "./simulations/output/${folder}/UI_${update_interval}" & 
    # > "./loginfo/${capacity}.txt" &
    done
done