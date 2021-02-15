import re
from utils import clean_string
from tqdm import tqdm
import os
import certifi_swisscom
from plato_client import CliTokenManager
from muesli_client import MuesliClient
import pandas as pd

jwt_token = {
    "access": "eyJ0eXAiOiJKV1QgYWNjZXNzIiwiYWxnIjoiUlMyNTYifQ.eyJqdGkiOiI1YWJjNzgwNzhiMWQ0YTRkOTYzOGRiMTMyOTRkY2IyOCIsImlzcyI6InBsYXRvLWNvcmUtYWFhLnNjYXBwLWNvcnAuc3dpc3Njb20uY29tIiwiZXhwIjoxNjEzMzMyMjgwLCJ1c2VyVHlwZSI6IlBsYXRvVXNlciIsInVzZXJJZCI6MzkyOCwicGVybWlzc2lvbnMiOnsicmVzb3VyY2VzIjp7IlRlbmFudCI6eyJwZXJtaXNzaW9ucyI6W10sIm9iamVjdFBlcm1pc3Npb25zIjp7ImFzciI6WyJ1c2UiLCJ1c2UiXSwiYXNyLWRhdGEiOlsidXNlIl0sImF6dXJlLXRyYWluaW5nIjpbInVzZSJdLCJjaXZyIjpbInVzZSIsInVzZSJdLCJjb3Ntb3MiOlsidXNlIl0sImUyZS10ZXN0aW5nIjpbInVzZSJdLCJuZW8iOlsidXNlIiwidXNlIl0sInN3aXNzY29tLXR2IjpbInVzZSIsInVzZSJdfX0sIkNvcnB1cyI6eyJwZXJtaXNzaW9ucyI6WyJyZWFkIiwibGFiZWxfcmVhZCIsIndyaXRlIiwiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwibGFiZWxfd3JpdGUiXSwib2JqZWN0UGVybWlzc2lvbnMiOnsiZXZhbC1zZXQtMyI6WyJyZWFkIiwibGFiZWxfcmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nIjpbInJlYWQiLCJsYWJlbF9yZWFkIiwicmVhZF9tYWNoaW5lIiwiY29udHJpYnV0b3JfcmVhZCIsInJlYWQiLCJsYWJlbF9yZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsInJlYWRfbWFjaGluZSJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctZ3N3IjpbIndyaXRlIiwicmVhZCIsImxhYmVsX3JlYWQiLCJjb250cmlidXRvcl9yZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1sYWJlbGxlZC1nc3ciOlsicmVhZCIsImxhYmVsX3JlYWQiLCJjb250cmlidXRvcl9yZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1hbmFseXNpcy1nc3ciOlsicmVhZCIsImxhYmVsX3JlYWQiLCJjb250cmlidXRvcl9yZWFkIiwicmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1hbmFseXNpcy1kZSI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIiwibGFiZWxfcmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nLWFuYWx5c2lzLWZyIjpbInJlYWQiLCJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCIsInJlYWQiLCJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctYW5hbHlzaXMtZW4iOlsicmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIiwicmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1hbmFseXNpcy1pdCI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiLCJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwibmVvLWNvbnZlcnNhdGlvbiI6WyJyZWFkIiwicmVhZF9tYWNoaW5lIiwid3JpdGUiLCJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiLCJyZWFkIiwicmVhZF9tYWNoaW5lIiwid3JpdGUiLCJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZGUtZ3N3LTEiOlsicmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIl0sInN3aXNzY29tLXR2LWJpd2Vla2x5LWRlLWdzdy0yIjpbInJlYWQiLCJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCJdLCJzd2lzc2NvbS10di1iaXdlZWtseS1kZS1nc3ctMyI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZnItMSI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZnItMiI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZnItMyI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZW4tMSI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZW4tMiI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktZW4tMyI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktaXQtMSI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktaXQtMiI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtYml3ZWVrbHktaXQtMyI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nLWdzdy0yIjpbInJlYWQiLCJjb250cmlidXRvcl9yZWFkIiwibGFiZWxfcmVhZCJdLCJzd2lzc2NvbS10di1yZWNvcmRpbmctZ3N3LTMiOlsicmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1nc3ctNCI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nLW11bHRpLWxhbmciOlsicmVhZCIsImNvbnRyaWJ1dG9yX3JlYWQiLCJsYWJlbF9yZWFkIl0sInN3aXNzY29tLXR2LXJlY29yZGluZy1nc3ctNSI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic3dpc3Njb20tdHYtcmVjb3JkaW5nLW11bHRpLWxhbmctMiI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwiY2l2ci1yZWNvcmRpbmctY2NpciI6WyJyZWFkIiwiY29udHJpYnV0b3JfcmVhZCIsImxhYmVsX3JlYWQiXSwic29mdC1kc3AtaW52ZXN0aWdhdGlvbiI6WyJyZWFkIiwid3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiXSwiZXZhbHNldC1iYWQtZmlsZXMiOlsicmVhZCIsIndyaXRlIiwibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIl0sImNpdnItcmVjb3JkaW5ncy1kZ3MtdHJhbnNmZXIiOlsicmVhZCIsInJlYWRfbWFjaGluZSIsIndyaXRlIiwiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIl0sImNpdnItcmVjb3JkaW5nLWNjaXItdHJhbnNsYXRpb24tYW5hbHlzaXMtMiI6WyJyZWFkIiwicmVhZF9tYWNoaW5lIiwid3JpdGUiLCJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiXSwiY2l2ci1yZWNvcmRpbmctY2Npci10cmFuc2xhdGlvbi1hbmFseXNpcy0xIjpbInJlYWQiLCJyZWFkX21hY2hpbmUiLCJ3cml0ZSIsImNvbnRyaWJ1dG9yX3JlYWQiLCJjb250cmlidXRvcl93cml0ZSIsImxhYmVsX3JlYWQiLCJsYWJlbF93cml0ZSJdLCJjaXZyLXJlY29yZGluZyI6WyJyZWFkIiwicmVhZF9tYWNoaW5lIiwid3JpdGUiLCJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiXSwiYXp1cmUtdHJhaW5pbmctcGlwZWxpbmUtY2l2ciI6WyJyZWFkIiwid3JpdGUiLCJjb250cmlidXRvcl9yZWFkIiwiY29udHJpYnV0b3Jfd3JpdGUiLCJsYWJlbF9yZWFkIiwibGFiZWxfd3JpdGUiXSwiYXp1cmUtdHJhaW5pbmctcGlwZWxpbmUtdHYiOlsicmVhZCIsIndyaXRlIiwiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIl0sImFzci1ldmFsdWF0aW9uLXJlcG9ydHMiOlsicmVhZCIsInJlYWRfbWFjaGluZSIsIndyaXRlIiwiY29udHJpYnV0b3JfcmVhZCIsImNvbnRyaWJ1dG9yX3dyaXRlIiwibGFiZWxfcmVhZCIsImxhYmVsX3dyaXRlIiwicmVhZCIsInJlYWRfbWFjaGluZSIsIndyaXRlIl0sInNhcy1kYXRhIjpbInJlYWQiLCJ3cml0ZSIsInJlYWRfbWFjaGluZSIsImNvbnRyaWJ1dG9yX3JlYWQiLCJjb250cmlidXRvcl93cml0ZSIsImxhYmVsX3JlYWQiLCJsYWJlbF93cml0ZSJdfX0sIlN0b3JhZ2VGb2xkZXIiOnsicGVybWlzc2lvbnMiOlsicmVhZCJdLCJvYmplY3RQZXJtaXNzaW9ucyI6eyJzd2lzc2NvbS10di1yZWNvcmRpbmciOlsicmVhZCIsInJlYWQiXSwiY2l2ci1yZWNvcmRpbmciOlsicmVhZCIsInJlYWRfbWFjaGluZSIsIndyaXRlIl0sImV2YWwtc2V0LTIiOlsicmVhZCJdLCJldmFsLXNldC0zIjpbInJlYWQiLCJyZWFkIl0sInNvZnQtZHNwLWludmVzdGlnYXRpb24iOlsicmVhZCIsIndyaXRlIl0sImV2YWxzZXQtYmFkLWZpbGVzIjpbInJlYWQiLCJ3cml0ZSJdLCJjaXZyLXJlY29yZGluZ3MtZGdzLXRyYW5zZmVyIjpbInJlYWQiLCJyZWFkX21hY2hpbmUiLCJ3cml0ZSJdLCJhenVyZS10cmFpbmluZy1waXBlbGluZS1jaXZyIjpbInJlYWQiLCJ3cml0ZSJdLCJhenVyZS10cmFpbmluZy1waXBlbGluZS10diI6WyJyZWFkIiwid3JpdGUiXSwiYXNyLWV2YWx1YXRpb24tcmVwb3J0cyI6WyJyZWFkIiwicmVhZCIsInJlYWRfbWFjaGluZSIsInJlYWRfbWFjaGluZSIsIndyaXRlIiwid3JpdGUiXSwic2FzLWRhdGEiOlsicmVhZCIsInJlYWRfbWFjaGluZSIsIndyaXRlIl19fSwiQ29uc2VudCI6eyJwZXJtaXNzaW9ucyI6W10sIm9iamVjdFBlcm1pc3Npb25zIjp7ImNpdnItcmVjb3JkaW5ncy1kZ3MtdHJhbnNmZXIiOlsicmVhZCIsIndyaXRlIl19fSwiQXNyTW9kZWwiOnsicGVybWlzc2lvbnMiOltdLCJvYmplY3RQZXJtaXNzaW9ucyI6e319LCJObHVTb2x1dGlvbiI6eyJwZXJtaXNzaW9ucyI6W10sIm9iamVjdFBlcm1pc3Npb25zIjp7fX0sIkxhYmVsaW5nQ2FtcGFpZ24iOnsicGVybWlzc2lvbnMiOltdLCJvYmplY3RQZXJtaXNzaW9ucyI6e319fX19.DmDeOg_Jqdh5ptm_JZx6CrrG9igQbwfs0TUK-vxYZT-EORzY8Wg_tBv1wxdKWM4Klor43tHVl-hBk7qGoq98Z9KSDPhP4miM90gLEVs15LkVqnereEIyt8Apf2C18RtnXJBJzYRXBNbd8ZmZUDjFXt2bSqy6fHclNPK1VTSZhRcZbHjM1sWh-t0seI__lvbSmmZ0zdCYsGlPVj_aFX8heLOWAN4Yf9tW15pBEDsz3IqcXqZclYQHrUo9EXaNNe0PGdx6t_1GcuOvWvo9ugmPI5mVXGOieeDnU7vZwTEPmWIGpdeDZkpZ_Q5MSU9KPNhhhO6PxMuSVp-g1nlhft5rXg",
    "refresh": "eyJ0eXAiOiJKV1QgcmVmcmVzaCIsImFsZyI6IlJTMjU2In0.eyJqdGkiOiJlOTRhYjY4MWE1MGI0OWNhOTQ5MWQ4YmU3NzQ0ZTkwNiIsImlzcyI6InBsYXRvLWNvcmUtYWFhLnNjYXBwLWNvcnAuc3dpc3Njb20uY29tIiwiZXhwIjoxNjEzMzY0NjgwLCJ1c2VyVHlwZSI6IlBsYXRvVXNlciIsInVzZXJJZCI6MzkyOH0.gZLLdJwxz4rJWWRJioOsMApZgqCFEBigCwKpgCkDwWB3D2r_1ilmEJjQmeOOWNzq-mqqejDyx1m5CT48O75sTO35GBxPKqnOT2SkXEyCYqVBF0OMktgIgVCLd2W0FomyMQ-YNMcHQEv610rFLtqFfv5xDQe1p_VUMAOwOh27pW4DIllJm7JN3L2Orw67JA2KAA3x3JHr2Osx7UBgkvTAoGnePptG_dACCpOknJGmMo0mgMo6qSW88K_CJi-3TfZKGUMPO2TUTw0faRs7M49dj95DkhFd44AIZ7_VUKOsC13UtTwBiGxfxJBc-uOyWYYRWG5hD9ig0t8h7PwC6kLQFw"
}


def norm_string(s):
    s = re.sub(r"\[\S*\]", '', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s


def filter_trian_datapoint(p):
    try:
        intent = \
        [x['content']['labels']['tv_intents'][0]['value'] for x in p['labels'] if x['type'] == 'human-labelling'][
            0]
    except:
        intent = 'Unk'

    try:
        entities_type = \
            [x['content']['labels']['tv_entities'][0]['type'] for x in p['labels'] if x['type'] == 'human-labelling'][0]
    except:
        entities_type = 'Unk'

    try:
        src_token = [x['metadata']['rawText'] for x in p['labels'] if x['type'] == 'human-labelling'][0]
    except:
        src_token = \
            [x['content']['transcriptions']['translation'] for x in p['labels'] if x['type'] == 'human-transcription'][
                0]
    if not src_token:
        src_token = \
            [x['content']['transcriptions']['transcription'] for x in p['labels'] if
             x['type'] == 'human-transcription'][0]

    language = [x['metadata']['language'] for x in p['labels'] if x['type'] == 'human-transcription'][0].lower()
    data = {
        'intent': intent,
        'language': language,
        'src': clean_string(norm_string(src_token)),
        'entities_type': entities_type
    }
    return data


# %%time
CA__CERT_PEM = certifi_swisscom.where()
os.environ['REQUESTS_CA_BUNDLE'] = CA__CERT_PEM
url = "https://plato-core-aaa.scapp-corp.swisscom.com"
token_manager = CliTokenManager(aaa_api_url=url, jwt_token=jwt_token)
client = MuesliClient('https://plato-core-muesli-backend.app.zhh.sbd.corproot.net', token_manager)

data_clean = []
for lang in ['de', 'en', 'fr', 'it']:
    main_corpus_data = client.get_all_datapoints(tenant='swisscom-tv',
                                                 corpus='swisscom-tv-recording-analysis-{}'.format(lang),
                                                 allow_human_transcription=True,
                                                 label_type_count=['human-transcription>0'])

    for p in tqdm(main_corpus_data):
        data_clean.append(filter_trian_datapoint(p))
output_path = './data/museli_analysis.json'
df = pd.DataFrame(data_clean)
df = df[df.src != '']
df = df[df.intent != 'Unk']
df = df[df.entities_type != 'Unk']
df = df[df.language.isin(['de', 'en', 'fr', 'it'])]
df.to_json(output_path, orient="records")
print(df.language.value_counts())