function t = build_tree(X,Y,cols)

% Create an empty decision tree, which has one node and everything in it
inds = {1:size(X,1)}; % A cell per node containing indices of all data in that node
p = 0; % Vector contiaining the index of the parent node for each node
labels = {}; % A label for each node
% q = [0 0 0 0]; % Vector containing [selected_feature best_feature] at each node

key = [];
feature = [];
left_child = [];
right_child = [];

% Create tree by splitting on the root
[inds p labels left_child right_child key feature] = split_node(X, Y, inds, p,
labels, cols, 1, left_child, right_child, key, feature);


t.inds = inds;
t.p = p;
t.labels = labels;
% t.q = q;

t.left_child = left_child;
t.right_child = right_child;
t.key = key;
t.feature = feature;

function [inds p labels left_child right_child key feature] = split_node(X, Y, inds, p, 
labels, cols, node, left_child, right_child, key, feature)
% Recursively splits nodes based on information gain

if node >= 15
    left_child(node) = 0;
    right_child(node) = 0;
    key(node) = 0;
    feature(node) = 0;    
    return;
end

% Check if the current leaf is consistent
if numel(unique(Y(inds{node}))) == 1
    left_child(node) = 0;
    right_child(node) = 0;
    key(node) = 0;
    feature(node) = 0;  
    return;
end

% Check if all inputs have the same features
% We do this by seeing if there are multiple unique rows of X
if size(unique(X(inds{node},:),'rows'),1) == 1
    left_child(node) = 0;
    right_child(node) = 0;
    key(node) = 0;
    feature(node) = 0;
    return;
end

% Otherwise, we need to split the current node on some feature

best_feature = 0; %best feature to split on
best_val = 0; % best value to split the best feature on
min_score = 1e6;

curr_X = X(inds{node},:);
curr_Y = Y(inds{node});
% Loop over each feature
for i = 1:size(X,2)     
    data_in = horzcat(curr_X, curr_Y);    
    [left_tree , right_tree, threshold] = split_data(data_in, i);    
    score = evaluate_split(left_tree, right_tree);
    
    if (score < min_score)
        min_score = score;
        best_val = threshold;
        best_feature = i;
    end
end

% Split the current node into two nodes
feat = curr_X(:,best_feature);
feat = feat < best_val;
inds = [inds; inds{node}(feat); inds{node}(~feat)];
inds{node} = [];
p = [p; node; node];
labels = [labels; sprintf('%s < %2.2f', cols{best_feature}, best_val); ...
    sprintf('%s >= %2.2f', cols{best_feature}, best_val)];

% Recurse on newly-create nodes
n = numel(p)-2;

left_child(node) = n+1;
right_child(node) = n+2;
key(node) = best_val;
feature(node) = best_feature;

[inds p labels left_child right_child key feature] = split_node(X, Y, inds, p, 
labels, cols, n+1, left_child, right_child, key, feature);
[inds p labels left_child right_child key feature] = split_node(X, Y, inds, p, 
labels, cols, n+2, left_child, right_child, key, feature);