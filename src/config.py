class DefaultConfig:
    MODEL_NAME = "klue/roberta-base"
    MODEL_SAVE_PATH = "./models/"
    LOG_SAVE_PATH = "./results/"
    DATA_SAVE_PATH = "./data/"
    DEMO_PATH = "./demo/"
    SEED = 2022
    
    OPTION = ""
    
    TRAIN_BATCH = 32
    DEV_BATCH = 64
    TEST_BATCH = 64
    
    TRAIN_LOG_INTERVAL = 1
    VALID_LOG_INTERVAL = 1
    
    KISTI_FILE_PATH = './data/tagging_train_result.json'
    VOCAB_FILE = "./data/vocab_kisti.txt"
    
    hierarchy = {
    '연구 목적': ['문제 정의', '가설 설정', '기술 정의'],
    '연구 방법': ['제안 방법', '대상 데이터', '데이터처리', '이론/모형'],
    '연구 결과': ['성능/효과', '후속연구']
    }
