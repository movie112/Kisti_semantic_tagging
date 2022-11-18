import os

if __name__ == '__main__':

    ### TRAINING ###
    """
    os.system("python train.py --epochs 1 \
                               --plm korscibert \
                               --use_section no_section \
                               --model_name test1.bin"
                               )
    os.system("python train.py --epochs 1 \
                               --plm korscibert \
                               --use_section raw_section \
                               --model_name test2.bin"
                               )  
    os.system("python train.py --epochs 1 \
                               --plm korscibert \
                               --use_section idx_section \
                               --model_name test3.bin"
                               )
    os.system("python train.py --epochs 1 \
                               --plm roberta \
                               --use_section no_section \
                               --model_name test4.bin"
                               ) 
    os.system("python train.py --epochs 1 \
                               --plm roberta \
                               --use_section raw_section \
                               --model_name test5.bin"
                               )  
    os.system("python train.py --epochs 1 \
                               --plm roberta \
                               --use_section idx_section \
                               --model_name test6.bin"
                               )
  
    
    # korscibert no section
    os.system("python inference.py --plm korscibert \
                                   --use_section no_section \
                                   --model_name ./models/korscibert_cosine_9050.bin \
                                   --loader_name ./data/korscibert_test_dataloader.pkl")
    # roberta no section
    os.system("python inference.py --plm roberta \
                                   --use_section no_section \
                                   --model_name ./models/roberta_cosine_9005.bin \
                                   --loader_name ./data/test_dataloader.pkl")
    """
    # korscibert raw section
    os.system("python inference.py --plm korscibert \
                                   --use_section raw_section \
                                   --model_name ./models/korscibert_raw.bin \
                                   --loader_name ./data/korscibert_osection_test_dataloader.pkl")
    # korscibert idx section
    os.system("python inference.py --plm korscibert \
                                   --use_section idx_section \
                                   --model_name ./models/korscibert_idx.bin \
                                   --loader_name ./data/korscibert_isection_test_dataloader.pkl")
