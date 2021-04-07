function [d] = select_traces(data, percent)
%  
% TO DO: make a mask to create the undersampling data
%
%  Input Parameters:
%       data:---------------------% full dataset structure
%       percent:------------------% percentage traces left
%     
%  Output Parameters:
%       d_un:---------------------% the undersampling data
%       mask_vec:-----------------% the selecting mask (vector)
%       mask_matrix:--------------% the selecting mask (matrix)
% 
%   Copyright:  Junhai Cao, 08-01-2019.
%   Email:      junhaicao1990@163.com/J.Cao@tudelft.nl
%   Place:      Department of Applied Physics, TU delft

nx = size(data,2); 	% the total traces number
nt = size(data,1);               % the total time samples

seismic_data_matrix = reshape(data, nt, nx); % seismic data maxtrix
mask_matrix = zeros(nt, nx); % make a mask to select the traces used
num_mask = floor(nx*percent/100); % the traces left

mask_index = sort(unique(randperm(nx, num_mask))); % make unique selection
mask_matrix(:, mask_index) = 1;

d = seismic_data_matrix.*mask_matrix;