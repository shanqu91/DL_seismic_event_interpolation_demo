
Folders:
smallsetdata_training/                        -> for small dataset training train data + dev data = 1000
fulldata_training/                            -> for large dataset training train data + dev data = 8000
lowslownessdata_training/                     -> for large dataset training train data + dev data = 8000, but excluding steeper events
test_data/                                    -> generate test data and predict the final data with all the trained model

In Delphi DEMO, we only run through the smallsetdata_training, for fulldata and lowslownessdata, we save the pre-trained models in Data/

in smallsetdata_training/ :
octave A_Generate_train_dev_Data_smallset.m & -> generate the original train+dev data,randomly mask traces to set the input/output train+dev data for training
python B_CNN_smallset.py                      -> training

in test_data/ :
octave A_Generate_test_Data.m &               -> generate test data, randomly mask traces to set the input test data
python B_predict_smallset.py                  -> predict the output test data with the model trained with small dataset
python B_predict_lowslowness.py               -> predict the output test data with the model trained with large dataset, but exclude steeper events
python B_predict_full.py                      -> predict the output test data with the model trained with large dataset, and include steeper events
octave C_check_predict_testdata.m &           -> if you want to plot output of test_data in matlab
