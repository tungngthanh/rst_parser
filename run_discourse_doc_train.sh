export DATA_PATH='./processed_data'
export MODE='train'
export FEAT='char'
export LEARNING_RATE_SCHEDULE='Exponential'
export PRETRAINED_EMBEDDING='data/glove.6B.100d.txt'
#export N_EMBED=300

export BATCH_SIZE=10000
python -m src.cmds.pointing_discourse train -b -d 0 -p exp/ptb.pointing.discourse.$FEAT \
--data_path $DATA_PATH -f $FEAT --learning_rate_schedule $LEARNING_RATE_SCHEDULE --batch-size $BATCH_SIZE --conf 'discourse_config.ini' \
--embed $PRETRAINED_EMBEDDING
#--n-embed $N_EMBED
