import re
from classes.command_generator.cleaning import clean_string
from tqdm import tqdm
import json
import os
import certifi_swisscom
from plato_client import CliTokenManager
from muesli_client import MuesliClient
from helper import read_data_json, add_pred
import pandas as pd

def dump_json(outfile, data, has_no_output):
    if data:
        if has_no_output:
            outfile.write(json.dumps(data))
        else:
            outfile.write(",")
            outfile.write(json.dumps(data))
        has_no_output = False
    return has_no_output

def norm_string(s):
    s = re.sub(r"\[\S*\]", '', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s
    
def filter_trian_datapoint(p):
    try:
        id_ =  [x['content']['labels']['tv_intents'][0]['value'] for x in p['labels'] if x['type'] == 'human-labelling'][0]
    except:
        id_ = 'Unk'
    
    try:
        entities_type = [x['content']['labels']['tv_entities'][0]['type'] for x in p['labels'] if x['type'] == 'human-labelling'][0]
    except:
        entities_type = 'Unk'
        
    
    try:
        src_token = [x['metadata']['rawText'] for x in p['labels'] if x['type'] == 'human-labelling'][0]
    except:
        src_token = [x['content']['transcriptions']['translation'] for x in p['labels'] if x['type'] == 'human-transcription'][0]
    if not src_token:
        src_token = [x['content']['transcriptions']['transcription'] for x in p['labels'] if x['type'] == 'human-transcription'][0]
        
    language = [x['metadata']['language'] for x in p['labels'] if x['type'] == 'human-transcription'][0].lower()
    tgt_token = [x['content']['prettifiedText'] for x in p['labels'] if x['type'] == 'human-labelling'][0]
    data = {
        'id': id_,
        'language': language,
        'src_token': clean_string(norm_string(src_token)),
        'tgt_token': clean_string(norm_string(tgt_token)),
        'entities_type': entities_type
    }
    return data



# %%time
CA__CERT_PEM =  certifi_swisscom.where()
os.environ['REQUESTS_CA_BUNDLE'] = CA__CERT_PEM
url = "https://plato-core-aaa.scapp-corp.swisscom.com"
jwt_token ={
  "access": "eyJ0eXAiOiJKV1QgYWNjZXNzIiwiYWxnIjoiUlMyNTYifQ.eyJqdGkiOiJhYzVhMTE1ZDIzODg0YjM1YTE3NDNlNjJjYjgyYjhmYyIsImlzcyI6InBsYXRvLWNvcmUtYWFhLnNjYXBwLWNvcnAuc3dpc3Njb20uY29tIiwiZXhwIjoxNjExNTM0Nzk3LCJ1c2VyVHlwZSI6IlBsYXRvVXNlciIsInVzZXJJZCI6MzkyOCwicGVybWlzc2lvbnMiOnsicmVzb3VyY2VzIjp7IlRlbmFudCI6eyJwZXJtaXNzaW9ucyI6W10sIm9iamVjdFBlcm1pc3Npb25zIjp7ImFzciI6WyJ1c2UiLCJ1c2UiXSwiYXNyLWRhdGEiOlsidXNlIl0sImF6dXJlLXRyYWluaW5nIjpbInVzZSJdLCJjaXZyIjpbInVzZSIsInVzZSJdLCJjb3Ntb3MiOlsidXNlIl0sImUyZS10ZXN0aW5nIjpbInVzZSJdLCJuZW8iOlsidXNlIiwidXNlIl0sInN3aXNzY29tLXR2IjpbInVzZSIsInVzZSJdfX0sIkNvcnB1cyI6eyJwZXJtaXNzaW9ucyI6WyJyZWFkIiwibGFiZWxfcmVhZCIsIndyaXRlIiwiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwibGFiZWxfd3JpdGUiXSwib2JqZWN0UGVybWlzc2lvbnMiOnsic3dpc3Njb20tdHYtcmVjb3JkaW5nIjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIiwicmVhZCIsInJlYWRfbWFjaGluZSIsInJlYWRfbWFjaGluZSJdLCJjaXZyLXJlY29yZGluZyI6WyJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiLCJyZWFkIiwicmVhZF9tYWNoaW5lIiwid3JpdGUiXSwibmVvLWNvbnZlcnNhdGlvbiI6WyJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIiwibGFiZWxfd3JpdGUiLCJyZWFkIiwicmVhZCIsInJlYWRfbWFjaGluZSIsInJlYWRfbWFjaGluZSIsIndyaXRlIiwid3JpdGUiXSwiZXZhbC1zZXQtMyI6WyJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJsYWJlbF9yZWFkIiwicmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nLWdzdyI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiLCJ3cml0ZSJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctbGFiZWxsZWQtZ3N3IjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIiwicmVhZCJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctYW5hbHlzaXMtZ3N3IjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIiwicmVhZCJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctYW5hbHlzaXMtZGUiOlsiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiLCJyZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1hbmFseXNpcy1mciI6WyJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJsYWJlbF9yZWFkIiwicmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nLWFuYWx5c2lzLWVuIjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIiwicmVhZCJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctYW5hbHlzaXMtaXQiOlsiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiLCJyZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1nc3ctMiI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZGUtZ3N3LTEiOlsiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIl0sInN3aXNzY29tLXR2LWJpd2Vla2x5LWRlLWdzdy0yIjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIiwicmVhZCJdLCJzd2lzc2NvbS10di1iaXdlZWtseS1kZS1nc3ctMyI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZnItMSI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZnItMiI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZnItMyI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZW4tMSI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZW4tMiI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZW4tMyI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktaXQtMSI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktaXQtMiI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktaXQtMyI6WyJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nLWdzdy0zIjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIiwicmVhZCJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctZ3N3LTQiOlsiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1tdWx0aS1sYW5nIjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIiwicmVhZCJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctZ3N3LTUiOlsiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1tdWx0aS1sYW5nLTIiOlsiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIl0sImNpdnItcmVjb3JkaW5nLWNjaXIiOlsiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIl0sInNvZnQtZHNwLWludmVzdGlnYXRpb24iOlsibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIiwicmVhZCIsIndyaXRlIl0sImNpdnItcmVjb3JkaW5nLWNjaXItdHJhbnNsYXRpb24tYW5hbHlzaXMtMSI6WyJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiLCJyZWFkIiwicmVhZF9tYWNoaW5lIiwid3JpdGUiXSwiY2l2ci1yZWNvcmRpbmctY2Npci10cmFuc2xhdGlvbi1hbmFseXNpcy0yIjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJjb250cmlidXRvcl93cml0ZSIsImxhYmVsX3JlYWQiLCJsYWJlbF93cml0ZSIsInJlYWQiLCJyZWFkX21hY2hpbmUiLCJ3cml0ZSJdLCJldmFsc2V0LWJhZC1maWxlcyI6WyJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiLCJyZWFkIiwid3JpdGUiXSwiY2l2ci1yZWNvcmRpbmdzLWRncy10cmFuc2ZlciI6WyJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiLCJyZWFkIiwicmVhZF9tYWNoaW5lIiwid3JpdGUiXSwiYXp1cmUtdHJhaW5pbmctcGlwZWxpbmUtY2l2ciI6WyJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiLCJyZWFkIiwid3JpdGUiXSwiYXp1cmUtdHJhaW5pbmctcGlwZWxpbmUtdHYiOlsiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIiwicmVhZCIsIndyaXRlIl0sImFzci1ldmFsdWF0aW9uLXJlcG9ydHMiOlsiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIiwicmVhZCIsInJlYWQiLCJyZWFkX21hY2hpbmUiLCJyZWFkX21hY2hpbmUiLCJ3cml0ZSIsIndyaXRlIl0sInNhcy1kYXRhIjpbImNvbnRyaWJ1dG9yX3JlYWQiLCJjb250cmlidXRvcl93cml0ZSIsImxhYmVsX3JlYWQiLCJsYWJlbF93cml0ZSIsInJlYWQiLCJyZWFkX21hY2hpbmUiLCJ3cml0ZSJdfX0sIlN0b3JhZ2VGb2xkZXIiOnsicGVybWlzc2lvbnMiOlsicmVhZCJdLCJvYmplY3RQZXJtaXNzaW9ucyI6eyJzd2lzc2NvbS10di1yZWNvcmRpbmciOlsicmVhZCIsInJlYWQiXSwiY2l2ci1yZWNvcmRpbmciOlsicmVhZCIsInJlYWRfbWFjaGluZSIsIndyaXRlIl0sImV2YWwtc2V0LTIiOlsicmVhZCJdLCJldmFsLXNldC0zIjpbInJlYWQiLCJyZWFkIl0sInNvZnQtZHNwLWludmVzdGlnYXRpb24iOlsicmVhZCIsIndyaXRlIl0sImV2YWxzZXQtYmFkLWZpbGVzIjpbInJlYWQiLCJ3cml0ZSJdLCJjaXZyLXJlY29yZGluZ3MtZGdzLXRyYW5zZmVyIjpbInJlYWQiLCJyZWFkX21hY2hpbmUiLCJ3cml0ZSJdLCJhenVyZS10cmFpbmluZy1waXBlbGluZS1jaXZyIjpbInJlYWQiLCJ3cml0ZSJdLCJhenVyZS10cmFpbmluZy1waXBlbGluZS10diI6WyJyZWFkIiwid3JpdGUiXSwiYXNyLWV2YWx1YXRpb24tcmVwb3J0cyI6WyJyZWFkIiwicmVhZCIsInJlYWRfbWFjaGluZSIsInJlYWRfbWFjaGluZSIsIndyaXRlIiwid3JpdGUiXSwic2FzLWRhdGEiOlsicmVhZCIsInJlYWRfbWFjaGluZSIsIndyaXRlIl19fSwiQ29uc2VudCI6eyJwZXJtaXNzaW9ucyI6W10sIm9iamVjdFBlcm1pc3Npb25zIjp7ImNpdnItcmVjb3JkaW5ncy1kZ3MtdHJhbnNmZXIiOlsicmVhZCIsIndyaXRlIl19fSwiQXNyTW9kZWwiOnsicGVybWlzc2lvbnMiOltdLCJvYmplY3RQZXJtaXNzaW9ucyI6e319LCJObHVTb2x1dGlvbiI6eyJwZXJtaXNzaW9ucyI6W10sIm9iamVjdFBlcm1pc3Npb25zIjp7fX0sIkxhYmVsaW5nQ2FtcGFpZ24iOnsicGVybWlzc2lvbnMiOltdLCJvYmplY3RQZXJtaXNzaW9ucyI6e319fX19.OcUrdoGoPxrd0LDQ9VqCWmQBzd4_h48FbFhoSG3hT8bRzTfiuX0cDEMJBJnoKK3QOtUIe2zL2qbNyXJMV9hk-sx_Cb9v1WpWs-K7K0_74X8qGFnUG-qpJh3rcuS4oYqHsEMCsOvAS0axiYwdRjlcANEm9GT23DcUsC5RfDY76qiep7UHa3P-6mHJ2yFFqRrKiDy1hP2u99fJWkNMQhf79vX8KfJra2Bhb3VkS6EZFM5yNJMq5c0_DlgoMl6zVaWpJXiP1mpUZxvLoWmUHWaeXXzxnYFwOP229HfWuNkU78RQzG1E4qwO-41bEkOnyIDk0Ho6kSBbRKNTqMML5NM_ng",
  "refresh": "eyJ0eXAiOiJKV1QgcmVmcmVzaCIsImFsZyI6IlJTMjU2In0.eyJqdGkiOiI4NTFmZmI4Y2I5NjY0ZGZjYjJlMDBkNzI5ZWRlZjI5ZiIsImlzcyI6InBsYXRvLWNvcmUtYWFhLnNjYXBwLWNvcnAuc3dpc3Njb20uY29tIiwiZXhwIjoxNjExNTY3MTk3LCJ1c2VyVHlwZSI6IlBsYXRvVXNlciIsInVzZXJJZCI6MzkyOH0.AXQm465UrdNSqHDadjn8eoIbE1FrUDtAcHaOiRnHKHQ3okf1ylokLxLwEfOa9Vx01hsmXT1vSKyRBrfZ8aNzoDbrZy43BSn2q7XDiXxM9-GV-k8sAqsihcPaeaCLhc93B2gzj190ImE2W-TskeOqdqAoWmGBOfED1uHp399dflNoUdcH3ge9rLcpYA7cY0_Lfk0qgYFg5rr4hfyjP8MlsYmBu5XpD_YSbOMY5qdetaYFUnVjngKyReTA95x2kGGQG6Y_ToWC6p6vzubdGa7wusOtiezkxc9uPdW82N4lg-H703hniDHxfcDxsFSin5eYHddT8FAzaEWvrabLWkaRQQ"
}
token_manager = CliTokenManager(aaa_api_url=url, jwt_token=jwt_token)
client = MuesliClient('https://plato-core-muesli-backend.app.zhh.sbd.corproot.net', token_manager)



for lang in ['de','en','fr','it']:
    output_path = 'analysis_{}.json'.format(lang)
    main_corpus_data = client.get_all_datapoints(tenant='swisscom-tv', 
                                              corpus='swisscom-tv-recording-analysis-{}'.format(lang), 
                                              allow_human_transcription=True,
                                              label_type_count=['human-transcription>0'])  
    data_clean = []
    for p in tqdm(main_corpus_data):
        data_clean.append(filter_trian_datapoint(p))

    df = pd.DataFrame(data_clean)
    df = df[df.src_token != '']
    df = df[df.tgt_token != '']
    df.to_json(output_path, orient="records")