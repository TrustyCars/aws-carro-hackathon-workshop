import editdistance
import pandas as pd
import os


def calculate_accuracy(result_df, ):
    
    PLATE_NUMBER_KEY = 'Plate Number'
    PREDICTED_TEXT_KEY = 'Predicted Text'

    result_df["CONCAT_PLATE_NUMBER_LEN"] = result_df[PLATE_NUMBER_KEY].str.len()
    result_df["editdistance"] = result_df.loc[:, [PLATE_NUMBER_KEY,PREDICTED_TEXT_KEY]].apply(lambda x: editdistance.eval(*x), axis=1)
    
    all_accuracy = (sum(result_df["CONCAT_PLATE_NUMBER_LEN"]) - sum(result_df["editdistance"])) / sum(result_df["CONCAT_PLATE_NUMBER_LEN"])
    cer = sum(result_df["editdistance"])/sum(result_df["CONCAT_PLATE_NUMBER_LEN"])
    normalized_cer = sum(result_df["editdistance"] / (sum(result_df["CONCAT_PLATE_NUMBER_LEN"]) + sum(result_df["editdistance"])))
    
    
    print ("All accuracy", all_accuracy * 100)
    
    print('CER', cer * 100)
    print('Normalized CER', normalized_cer * 100)
    print("Normalized Accuraccy", 100 - (normalized_cer * 100))
    
    return result_df

