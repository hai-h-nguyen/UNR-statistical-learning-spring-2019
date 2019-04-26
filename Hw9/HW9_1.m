clear all;
close all;
clc;

%% Load the auto data
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

X = train_data(:, 1:end-1);
Y = train_data(:, end);

cols = {'Hits', 'Runs', 'RBI', 'Walks', 'Years','PutOuts'};

%% Build the decision tree
t = build_tree(X,Y,cols);

%% Display the tree
treeplot(t.p');
title('Decision tree');
[xs,ys,h,s] = treelayout(t.p');

for i = 2:numel(t.p)
  % Get my coordinate
  my_x = xs(i);
  my_y = ys(i);

  % Get parent coordinate
  parent_x = xs(t.p(i));
  parent_y = ys(t.p(i));

  % Calculate weight coordinate (midpoint)
  mid_x = (my_x + parent_x)/2;
  mid_y = (my_y + parent_y)/2;

    % Edge label
  text(mid_x,mid_y,t.labels{i-1});
    
    % Leaf label
    if ~isempty(t.inds{i})
        val = Y(t.inds{i});
        text(my_x, my_y, sprintf('y=%2.2f\nn=%d', mean(val), numel(val)));
    end
end

%% Prediction
predicted_Y = zeros(size(test_data,1), 1);

%% calculate mean value for each leaf
for j = 1 : size(t.inds,1)
    idx_arr = cell2mat(t.inds(j));
    if (~isempty(idx_arr))
        t.mean_predict(j) = mean(Y(idx_arr));
    else
        t.mean_predict(j) = 0;
    end
end
 
for i = 1 : size(test_data,1)    
    % predict from input
    X_test = test_data(i, 1:end-1);
    node = 1;
    
    % Go down the tree
    while ((t.left_child(node) > 0) && (t.right_child(node) > 0)) % not a leaf
        if (X_test(t.feature(node)) < t.key(node))
            node = t.left_child(node);
        else
            node = t.right_child(node);
        end
    end
    
    predicted_Y(i) = t.mean_predict(node);       
end
MSE_test = sqrt(sum((predicted_Y - test_data(i, end)).^2) / size(test_data,1));
fprintf('Test MSE %f', MSE_test);