function [d] = select_traces_biggap(data, percent, gap)
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
mask_matrix = ones(nt, nx); % make a mask to select the traces empty
num_mask = floor(nx*(100-percent)/100/gap); % the traces left

mask_index = sort(unique(randperm(floor(nx/gap), num_mask))); % make unique selection
mask_index = mask_index .* gap;

for ix = 1 : nx
    for i_index = 1 : num_mask
        if ix == mask_index(i_index)
            mask_matrix(:, ix) = 0;
            for igap = 1 : gap
                if (ix+igap)<=nx
                    mask_matrix(:, ix+igap) = 0;
                end
            end
        end
    end
end
d = seismic_data_matrix.*mask_matrix;