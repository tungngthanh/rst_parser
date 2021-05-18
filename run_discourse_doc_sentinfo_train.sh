export DATA_PATH='./processed_data'
export MODE='train'
export FEAT='bert'
export LEARNING_RATE_SCHEDULE='Exponential'
export PRETRAINED_EMBEDDING='data/glove.6B.100d.txt'
#export N_EMBED=300

export BERT_MODEL='xlnet-base-cased'
export BATCH_SIZE=4000
python -m src.cmds.pointing_discourse_sentinfo train -b -d 0 -p exp/ptb.pointing.discourse.sent_info.$FEAT \
--data_path $DATA_PATH -f $FEAT --learning_rate_schedule $LEARNING_RATE_SCHEDULE \
--bert $BERT_MODEL --batch-size $BATCH_SIZE --conf 'discourse_config_bert.ini'
#--embed $PRETRAINED_EMBEDDING
#--n-embed $N_EMBED