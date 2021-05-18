export BATCH_SIZE=4000
export TEST_FILE='dummy_format_data/sample_full_data_format'
export PREDICT_PATH='saved_model/model_pointing_discourse_sent_info_Full_RNF_46.7.pt'
export PREDICT_OUTPUT_PATH='dummy_format_data/sample_pred_file_sent_info'

python -m src.cmds.pointing_discourse_sentinfo predict -b -d 0 -p exp/ptb.pointing.discourse.sent_info.bert.predict \
--data $TEST_FILE --path $PREDICT_PATH --batch-size $BATCH_SIZE --predict_output_path $PREDICT_OUTPUT_PATH