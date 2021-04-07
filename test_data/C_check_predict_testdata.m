clc;clear;close all;
load ../Data/predicted_test_data_lowslowness.mat
load ../Data/X_test.mat

figure;imagesc([squeeze(X_test)  squeeze(Y_test)]);
title('with 8000 data excluding steep events, test data (left->input, right->predicted)')
print -djpg ../Fig/predicted_test_data_lowslowness.jpg

load ../Data/predicted_test_data_full.mat
figure;imagesc([squeeze(X_test)  squeeze(Y_test)]);
title('with 8000 full data including steep events, test data (left->input, right->predicted)')
print -djpg ../Fig/predicted_test_data_full.jpg

load ../Data/predicted_test_data_smallset.mat
figure;imagesc([squeeze(X_test)  squeeze(Y_test)]);
title('with 1000 data including steep events, test data (left->input, right->predicted)')
print -djpg ../Fig/predicted_test_data_smallset.jpg

waitfor(gcf);




