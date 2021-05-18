export TEST_FILE='dummy_format_data/sample_full_data_format'
export PREDICT_PATH='saved_model/model_pointing_discourse_gold_segmentation_edu_rep_Full_RNF_51.1.pt'
export PREDICT_OUTPUT_PATH='dummy_format_data/sample_pred_file_gold_segmentation_edu_rep'
export BATCH_SIZE=4000
export BEAM_SIZE=20

python -m src.cmds.pointing_discourse_gold_segmentation_edu_rep predict -b -d 0 --data $TEST_FILE \
--path $PREDICT_PATH --batch-size $BATCH_SIZE --beam-size $BEAM_SIZE --predict_output_path $PREDICT_OUTPUT_PATH