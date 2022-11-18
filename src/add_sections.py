import pandas as pd
import re
import json
import ijson
from tqdm import tqdm
import numpy as np


def load_data(file_path, data_type='original', doc_ids=[]): 
    """
    문장의미태깅 또는 논문전문 원본데이터 로드 함수

        Args:
            file_path (str) : 데이터 파일이 저장된 경로
            data_type (str) : 'original' or 'full'
            doc_ids (list) : data_type이 'full'인 경우, 가져올 논문 'doc_id' 리스트
        Retruns:
            data (DataFrame) : 데이터 파일로부터 DataFrame형식으로 데이터 로드한 결과
    """

    if data_type == 'original':
        print("============== 문장 의미 태깅 데이터셋 로드 ============")
        # 변수 초기화
        data = pd.DataFrame(columns=['doc_id', 'sentence', 'tag', 'keysentence'])
        fine2coarse = {'문제 정의':'연구 목적', '가설 설정':'연구 목적', '기술 정의':'연구 목적', \
                        '제안 방법':'연구 방법', '대상 데이터':'연구 방법', '분석방법':'연구 방법', '데이터처리':'연구 방법', \
                        '이론/모형':'연구 방법', '성능/효과':'연구 결과', '후속연구':'연구 결과'}
        
        # 원본데이터를 DataFrame에 저장
        with open(file_path, "r", encoding="utf8") as json_file:
            for index, json_line in enumerate(tqdm(json_file, desc='read_data')):
                json_data = json.loads(json_line)
                row = pd.json_normalize(json_data)
                data = data.append(row)

        # 'keysentence' 칼럼 제거
        data.drop(columns="keysentence", inplace=True)

        # 대분류 칼럼 'root_tag' 추가
        data['root_tag'] = data['tag'].apply(lambda x: fine2coarse[x])

        # 명시된 레이블이 아닌 태그 '데이터 처리'는 '분석 방법'으로 변경
        data['tag'] = data['tag'].replace({'데이터처리':'분석방법'})


    if data_type == 'full':
        print("============== 논문 전문 데이터셋 로드 ============")
        # 변수 초기화
        data = pd.DataFrame(columns=['doc_id', 'body_text'])
        N = 0

        with open(file_path, "rb") as f:
            for record in tqdm(ijson.items(f, "item")):
                row = []
                doc_id = record["doc_id"]
                if doc_id not in doc_ids:
                    continue

                # title_ko = ""
                # title_en = ""
                # if "ko" in record["title"].keys():
                #     title_ko = record["title"]["ko"]
                # if "en" in record["title"].keys():
                #     title_en = record["title"]["en"]
                # authors = record["authors"]
                # journal_ko = ""
                # journal_en = ""
                # if "ko" in record["journal"].keys():
                #     journal_ko = record["journal"]["ko"]
                # if "en" in record["journal"].keys():
                #     journal_en = record["journal"]["en"]
                # year = record["year"]
                # abstract = {}
                # if "abstract" in record.keys():
                #     abstract = record["abstract"]      # type: dict
                # keywords = {}
                # if "keywords" in record.keys():
                #     keywords = record["keywords"]      # type: dict
                body_text = record["body_text"]      # type: list, element: dict
                
                row.append(doc_id)
                # row.append(title_ko)
                # row.append(title_en)
                # row.append(authors)
                # row.append(journal_ko)
                # row.append(journal_en)
                # row.append(year)
                # row.append(abstract)
                # row.append(keywords)
                row.append(body_text)

                data.loc[N] = row

                N += 1

    # 'doc_id' 칼럼 기준 오름차순 정렬
    data.sort_values(by='doc_id', axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    print("데이터 로드 완료")
     
    return data


def add_section_from_fullpaper(data, file_path, doc_ids):
    """
    논문 전문 데이터셋을 사용하여 의미 태깅 문장 데이터셋에 있는 문장이 해당하는 section명을 가져오는 함수입니다.

    section명을 찾기 위해 "의미 태깅 문장"과 "논문 전문"에 적용한 전처리 기법
    1) 공백, 개행문자 제거
    2) 영어 소문자 변환
    3) html 태그 제거
    4) 한글, 영어, 숫자 외 문자 제거
    5) 소괄호 및 소괄호 내 내용 제거
    6) 숫자 포함 전후 문자 3개 제거
    7) "의미 태깅 문장"에 section명이 앞에 포함된 경우 section명 제거
    8) 4)를 포함하여 위의 방법들로 전처리하였을때 section명을 찾지 못한 경우, 한글 외 문자 제거
    

    round를 거듭할수록 더 많은 전처리가 적용됩니다.

        Args:
            data (DataFrame) : 문장 의미 태깅 데이터셋
            file_path (str) : 논문 전문 데이터셋이 저장된 파일 경로
            doc_ids (list) : 논문 전문 데이터셋에서 가져올 논문 'doc_id' 리스트
        Returns:
            data (DataFrame)
    """

    data_full = load_data(file_path, data_type='full', doc_ids=doc_ids)
    data_full = data_full[['doc_id', 'body_text']]

    doc_ids_full = list(data_full['doc_id'].unique())
    n_round = 4

    # print(len(doc_ids_full))

    for rounds in range(1, n_round+1):
        print("=========== round "+str(rounds)+" ===========")
        for docid in tqdm(doc_ids_full):
            body_text = data_full[data_full['doc_id']==docid]['body_text'].values[0]    # type: list, element: dict
            # section으로 text를, text로 section을 찾는 딕셔너리를 각각 생성
            section2text = {}
            text2section = {}
            section2idx = {}
            section_idx = 0
            for st_dict in body_text:
                # section 키 없는 경우
                if 'section' not in st_dict.keys():
                    text = ' '.join(st_dict['text'])
                    section2text['None'] = text
                    text2section[text] = 'None'
                # section, text 키 모두 있는 경우
                elif 'text' in st_dict.keys():
                    text = ' '.join(st_dict['text'])
                    # 같은 section명에 대한 text가 이미 있는 경우, 원래 있던 text와 concat
                    # text2section의 키는 따로 저장됨
                    if st_dict['section'] in section2text.keys():
                        section2text[st_dict['section']] = section2text[st_dict['section']] + ' ' + text
                        text2section[text] = st_dict['section']
                        section2idx[st_dict['section']] = section_idx
                    else:
                        section2text[st_dict['section']] = text
                        text2section[text] = st_dict['section']
                        section2idx[st_dict['section']] = section_idx
                    section_idx += 1
                # text 키 없는 경우
                else:
                    section2text[st_dict['section']] = ''
                    section2idx[st_dict['section']] = section_idx
                    section_idx += 1
            
            n_section = section_idx
            for k, v in section2idx.items():
                section2idx[k] = round(v/n_section, 3)
            
            
            # 해당 문서 전체 본문 리스트
            doc = list(text2section.keys())

            # 현재 doc_id에 해당하는 문장들로 이루어진 dataframe의 인덱스
            df_index = data[data['doc_id']==docid].index
            for i in df_index:        
                # 'section'칼럼에 이미 값이 있으면 패쓰
                if rounds > 1:
                    if not pd.isna(data.loc[i, 'body_text']):
                        continue
                
                target_s = data.loc[i, 'sentence']
                

                if rounds > 3:
                    for sec in text2section.values():                              # 8. 
                        if target_s.startswith(sec):
                            target_s = target_s.replace(sec, '')
                            break
                    
                target_s = re.sub(r'[\s]+', '', target_s)                           # 1) 공백, 개행문자 등 모두제거
                target_s = target_s.lower()                                         # 2) 영어는 모두 소문자로 변환
                target_s = re.sub(r'&lt;|&gt;', '', target_s)                       # 3) html태그 제거(&lt; &gt;)

                if rounds > 1:
                    target_s = re.sub(r'[^0-9a-zA-Z가-힣]+', '', target_s)             # 4. 숫자, 영어, 한글 외 문자 제거
                    target_s = re.sub(r'\([^\)]*\)', '', target_s)                     # 5. 소괄호와 그 안의 내용 제거
                if rounds > 2:
                    target_s = re.sub(r'.{3}[0-9]+.{3}', '', target_s)                 # 7. 숫자 양쪽 문자 3개 제거

                for part in doc:
                    part_flat = re.sub(r'[\s]+', '', part)                         # 1. 문자열 비교를 위해 공백, 개행문자 등 모두제거
                    part_flat = part_flat.lower()                                  # 2. 영어는 모두 소문자로 변환
                    part_flat = re.sub(r'&lt;|&gt;', '', part_flat)                # 3. html태그 제거(&lt; &gt;)
                    if rounds > 1:
                        part_flat = re.sub(r'[^0-9a-zA-Z가-힣]+', '', part_flat)       # 4. 숫자, 영어, 한글 외 문자 제거
                        part_flat = re.sub(r'\([^\)]*\)', '', part_flat)               # 5. 소괄호와 그 안의 내용 제거
                    if rounds > 2:
                        part_flat = re.sub(r'.{3}[0-9]+.{3}', '', part_flat)           # 7. 
                    
                    if target_s in part_flat:    # 문장이 본문에 포함되어있는지 확인
                        if text2section[part] == 'None' or text2section[part] == '':
                            data.loc[i, 'section'] = np.NaN
                        else:
                            data.loc[i, 'section'] = text2section[part]
                            data.loc[i, 's_idx'] = section2idx[text2section[part]]
                        data.loc[i, 'body_text'] = part
                        break
                
                # 만약 일치 문자열을 찾지 못한 경우 영어,숫자 제거 후 다시 비교
                target_s = re.sub(r'[a-zA-Z0-9]', '', target_s)                    # 6. 한글 외 문자 제거
                        
                for part in doc:
                    part_flat = re.sub(r'[\s]+', '', part)                         # 1. 문자열 비교를 위해 공백, 개행문자 등 모두제거
                    part_flat = part_flat.lower()                                  # 2. 영어는 모두 소문자로 변환
                    part_flat = re.sub(r'&lt;|&gt;', '', part_flat)                # 3. html태그 제거(&lt; &gt;)
                    part_flat = re.sub(r'[^가-힣]+', '', part_flat)                # 6. 한글 외 문자 제거
                    if rounds > 1:
                        part_flat = re.sub(r'\([^\)]*\)', '', part_flat)           # 5. 소괄호와 그 안의 내용 제거
                    if rounds > 2:
                        part_flat = re.sub(r'.{3}[0-9]+.{3}', '', part_flat)       # 7.
                    if target_s in part_flat:    # 문장이 본문에 포함되어있는지 확인
                        if text2section[part] == 'None' or text2section[part] == '':
                            data.loc[i, 'section'] = np.NaN
                        else:
                            data.loc[i, 'section'] = text2section[part]
                            data.loc[i, 's_idx'] = section2idx[text2section[part]]
                        data.loc[i, 'body_text'] = part
                        break
        
        print("section이 없는 행의 개수: ", data['body_text'].isnull().sum())
        
    return data


def save_as_pickle(data, file_path):
    """
    DataFrame을 pickle파일로 저장하는 함수
        Args:
            data (DataFrame) : 
            file_path (str) : 파일 저장 경로
        Returns:
            None
    """
    data.to_pickle(file_path)
    return


if __name__ == "__main__":
    # print(os.getcwd())
    # KISTI 데이터셋 경로
    ORIG_DATA_FILE_PATH = './data/tagging_train_result.json'
    FULL_DATA_FILE_PATH = './data/fullpaper.json'
    FINAL_DATA_FILE_PATH = './data/data_final_sidx.pkl'

    data = load_data(ORIG_DATA_FILE_PATH)
    doc_ids = list(data['doc_id'].unique())

    data_with_section = add_section_from_fullpaper(data, FULL_DATA_FILE_PATH, doc_ids=doc_ids)
    save_as_pickle(data_with_section, FINAL_DATA_FILE_PATH)