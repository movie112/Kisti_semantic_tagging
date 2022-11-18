import torch
import tokenization_kisti as tokenization

from utils import *
from preprocessing import *
from models import *

from transformers import logging
logging.set_verbosity_error()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_results(model, sentences, tags):

    max_length = 256
    tokenizer = tokenization.FullTokenizer(vocab_file="./data/vocab_kisti.txt", do_lower_case=False, tokenizer_type="Mecab")
    inputs_dict = {}

    coarse_labels = tags[0]
    coarse_preds = []

    fine_labels = tags[1]
    fine_preds = []

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens
        tokens = tokens[:max_length-1]
        tokens = tokens + ['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) <= max_length

        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_length - len(input_ids))

        input_ids = input_ids + padding
        attention_mask = attention_mask + padding

        inputs_dict['input_ids'] = torch.tensor([input_ids]).to(device)
        inputs_dict['attention_mask'] = torch.tensor([attention_mask]).to(device)


        coarse, fine, outputs, at1, at2 = model(input_ids=inputs_dict['input_ids'], attention_mask=inputs_dict['attention_mask'])

        coarse_label = int(torch.argmax(coarse, dim=-1)[0])
        fine_label = int(torch.argmax(fine, dim=-1)[0])

        coarse_dict = {0: '연구 목적',
                       1: '연구 방법',
                       2: '연구 결과'}
        fine_dict = {0: '문제 정의', 1: '가설 설정', 2: '기술 정의', 3: '제안 방법', 4: '대상 데이터',
                     5: '데이터처리', 6: '이론/모형', 7: '성능/효과', 8: '후속연구'}

        coarse_preds.append(coarse_dict[coarse_label])
        fine_preds.append(fine_dict[fine_label])

    for i in range(len(sentences)):
        logger.info("I N P U T : {}".format(sentences[i]))
        logger.info("PREDICTION: {},{}".format(coarse_preds[i], fine_preds[i]))
        logger.info("L A B E L : {},{}".format(coarse_labels[i], fine_labels[i]))
        logger.info("\n")

        print("I N P U T : {}".format(sentences[i]))
        print("PREDICTION: {},{}".format(coarse_preds[i], fine_preds[i]))
        print("L A B E L : {},{}".format(coarse_labels[i], fine_labels[i]))
        print("\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_file', type=str, default='./results/results.log', help='results')
    parser.add_argument('--logger', type=str, default='results')
    parser.add_argument('--plm', type=str, default='korscibert')

    sentences = ['본 실험에서 사용한 밸런스 핸들 장치는 FMA의 신전·굴곡 운동 평가 항목 (가장 상지 동작의 기본이 되는 운동)을 이용하여, 사용자의 상지 동작 시 반응하는 활성화 근육과 편마비 환자가 상지 재활동작 시 재활정도를 평가하는 활성화 근육과의 관련성을 비교하였다[24].',
 '마지막으로 레일 압력 1.5bar에서 실제 기관에서 작동하는 상태를 가정하여 구동 속도를 변화시키면서 4개의 인젝터를 순차분사(sequential injection)하였다(동적 분사).',
 '따라서 건강을 추구하는 소비자들에게는 디저트카페 만의 유기농 또는 신선한 제철 식재료로 만든 디저트를 제공하고, 특히 달콤하고 칼로리가 높아 다이어트 하는 여성 들이 기피하는 디저트가 아닌 달콤하고 맛있지만, 칼로리가낮아 오히려 건강에 도움이 되는 건강메뉴를 개발하여 제공한다면 디저트카페만의 성공적인 상품 전략에 도움이 될 수있을 것이다.',
 '혈액투석 환자의 영양장애에 영향을 미치는 요인은 단계별 다중 회귀분석(stepwise multiple regression)을 이용하여 분석하였다.\r\n\r\n',
 '또한 소프트웨어적으로는 위상 변이 제어를 통해 4 leg일 경우 90°, 2 leg일 경우 180°로 제어함으로서 leg 고장 시에 안정적으로 운전함을 확인하였다.',
 '식도 조직에서 western blot을 실시하여 염증성 매개인자인 p-IκBα와 NF-κBp65의 발현을 측정한 결과 AC100군에서 유의한 감소가 나타났으며, 이로 인해 염증성 사이토카인인 COX-2, iNOS, TNF-α 발현 또한 현저히 감소하였다.',
 '전사체(transcriptome)는 생명정보가 DNA에서 RNA로 전달된 전사물(transcript)의 총합을 의미하고, 전사물은 mRNA, tRNA 및 non-coding RNA 등 RNA의 모든 것을 포함하며, 전사체 프로파일은 RNA의 염기서열을 모두 읽어내는 RNAseq(RNA sequencing) 분석법을 이용하여 확보한다.',
 '본 연구는 종합병원 간호사의 다문화 가정에 대한 인식을 파악하여, 간호사들이 편견 없이 다문화 가정을 받아들이고, 효과적인 간호와 보건의료서비스를 제공하기 위한 기본적인 바탕과 개선방안을 마련하는데 기초자료를 제공하기 위하여 시도되었다.',
 '다만 본 연구에서 사용한 제 5차 학업중단 청소년 패널조사는 자가보고식으로 측정하여 학교 밖 청소년의 심리 ·\u200b\u200b\u200b\u200b\u200b\u200b\u200b 정서적 경험을 객관적이거나 심층적으로 측정하는 것에는 한계가 있으므로 질적연구 및 다층적인 방법을 통한 추후 연구를 제언한다.',
 '특히, 그래프-컷 알고리즘에 대한 불일치 레이블과 교합 처리의 효과를 실험적으로 평가한다.',
 '기업가적 지향성에 관한 정의로 Lumpkin & Dess(1996)는 다른 기업과 차별될 수 있는 혁신적인 서비스와 이를 개발하고자 하는 행동 또는 의사결정 프로세스를 기업가 지향성(entrepreneurial orientation)이라고 하였다.',
 '탈진할 때까지 수영 운동을 시켜 지구력과 혈액 내 피로물질들의 변화를 측정하고, 혈액과 근육조직의 효소활성을 측정하였다.',
 '진정성 리더십이란 지도자가 구성원들과 함께 일하면서 자아인식, 내재화된 도덕적 관점, 균형 잡힌 정보처리 및 관계적 투명성의 4개 하위영역을 통해 구성원들의 발전을 촉진하도록 하는 행동 유형이다[3][12].',
 '둘째, 사회적 지지와 가족기능 간의 관계를 장애수용이 매개하는지 검증한 결과 사회적 지지와 가족기능의 관계에서 장애수용은 부분적으로 매개하는 것으로 나타났다.',
 '(3)척력의 경우도본 논문에서는 근거리 척력에 해당하는 010 클러스트를적용하였으나, 실험상에서의 척력과 관련한 다양한 현상 들을 설명하기 위해서는 원거리 척력에 해당하는 클러스트 등의 도입도 필요하다. ']

    tag_0s = ['연구 방법',
 '연구 방법',
 '연구 결과',
 '연구 방법',
 '연구 결과',
 '연구 결과',
 '연구 목적',
 '연구 목적',
 '연구 결과',
 '연구 방법',
 '연구 목적',
 '연구 방법',
 '연구 목적',
 '연구 결과',
 '연구 결과']
    tag_1s = ['데이터처리',
 '제안 방법',
 '후속연구',
 '데이터처리',
 '성능/효과',
 '성능/효과',
 '기술 정의',
 '문제 정의',
 '후속연구',
 '데이터처리',
 '기술 정의',
 '제안 방법',
 '기술 정의',
 '성능/효과',
 '후속연구']

    tags = tag_0s, tag_1s

    cfg = DefaultConfig()
    args = parser.parse_args()
    logger = request_logger(f'{args.logger}', cfg, args)

    model = LSTMBertModel(cfg, args)
    model.to(device)
    model.parameters
    model.load_state_dict(torch.load('/data/kisti/models/korscibert_cosine_9050.bin', map_location=device))

    show_results(model, sentences, tags)
