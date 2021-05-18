#this is to create a pickle file with only input of raw text
#it would create dummy edu_break and docs_structure
import pickle
input_file="processed_data/test_approach1"
output_file="dummy_format_data/rawtext_data_format"
data_dict = pickle.load(open(input_file, "rb"))
sents= data_dict['InputDocs']
# final_output={'InputDocs':sents, 'EduBreak_TokenLevel':[], 'Docs_structure':[]}
# dummy_edu_breaks=[[len(sent)-1] for sent in sents]
# final_output['EduBreak_TokenLevel']=dummy_edu_breaks
# dummy_gold_metrics=[['NONE'] for i in range(len(sents))]
# final_output['Docs_structure']=dummy_gold_metrics
# with open(output_file, 'wb') as f:
# 	pickle.dump(final_output, f)
sample_output_file="dummy_format_data/sample_rawtext_data_format"
sample_output={'InputDocs':sents[:1], 'EduBreak_TokenLevel':[[len(sents[0])-1]], 'Docs_structure':[['NONE']]}
with open(sample_output_file, 'wb') as f:
	pickle.dump(sample_output, f)

sample_full_output_file="dummy_format_data/sample_full_data_format"
sample_full_output={'InputDocs':data_dict['InputDocs'][:1],
                    'EduBreak_TokenLevel':data_dict['EduBreak_TokenLevel'][:1],
                    'Docs_structure':data_dict['Docs_structure'][:1],
                    'SentBreak':data_dict['SentBreak'][:1]
                    }
with open(sample_full_output_file, 'wb') as f:
	pickle.dump(sample_full_output, f)
