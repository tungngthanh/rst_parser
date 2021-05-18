export BATCH_SIZE=4000
export TEST_FILE='dummy_format_data/sample_rawtext_data_format'
export PREDICT_PATH='saved_model/model_pointing_discourse_Full_RNF_44.4.pt'
export PREDICT_OUTPUT_PATH='dummy_format_data/sample_pred_file'

python -m src.cmds.pointing_discourse predict -b -d 0 -p exp/ptb.pointing.discourse.bert \
--data $TEST_FILE --path $PREDICT_PATH --batch-size $BATCH_SIZE --predict_output_path $PREDICT_OUTPUT_PATH