clc;
clear all;
data = readtable("Hitters.csv");

first_row = 0;

% Remove N/A salary
for i=1:size(data, 1)
    if (data{i, 20} == "NA")
        if first_row == 0
            row_removes = uint16(i);
            first_row = 1;
        else
            row_removes = horzcat(row_removes, i);
        end
    end
end

data(row_removes, :) = [];

% Use only important features and convert to log salary
new_data = table2array(data(:, [3 5 6 7 8 17]));
new_data = horzcat(new_data, log(str2double(data{:,20})));

% Randomly generate training and test set
% Randomly shuffle samples (rows)
total_sample = size(new_data, 1);
shuffledRowIdx = randperm(total_sample);
new_data = new_data(shuffledRowIdx, :);

% Training data
half = ceil(total_sample / 2);
train_data = new_data(1:half, :);
test_data = new_data(half + 1 : end, :);

% Create a full tree
num_features = size(train_data, 2) - 1;

score_list = [];

for i = 1:num_features
   % Calculate the mean of the features
   mean_feature = mean(train_data(:, i));
   
   % Pick this feature and perform the splitting around the mean value
   [left, right] = split_data(train_data, i);
    
   % Evaluate this split
   score = evaluate_split(left, right);
    
   score_list(i) = score;
end

% Pick the minimum from score_list and perform the split
[~, chosen_feature] = min(score_list);

[left, right] = split_data(train_data, chosen_feature);

% Perform this routine recursively on the left and right tree

% TODO: Save the tree structure to plot

% TODO: Stop condition
