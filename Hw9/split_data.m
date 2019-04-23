function [left_tree , right_tree] = split_data(data_in, chosen_feature)

    threshold = mean(data_in(:, chosen_feature));
   
    left_rows = find(data_in(:, chosen_feature) >= threshold);
    right_rows = find(data_in(:, chosen_feature) < threshold);
    
    left_tree = data_in(left_rows, :);
    right_tree = data_in(right_rows, :);
end
