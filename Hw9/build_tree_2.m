function t = build_tree_2(X,Y,cols)
% Builds a decision tree to predict Y from X.  The tree is grown by
% recursively splitting each node using the feature which gives the best
% information gain until the leaf is consistent or all inputs have the same
% feature values.
%
% X is an nxm matrix, where n is the number of points and m is the
% number of features.
% Y is an nx1 vector of classes
% cols is a cell-vector of labels for each feature
%
% RETURNS t, a structure with three entries:
% t.p is a vector with the index of each node's parent node
% t.inds is the rows of X in each node (non-empty only for leaves)
% t.labels is a vector of labels showing the decision that was made to get
%     to that node

% Create an empty decision tree, which has one node and everything in it
inds = {1:size(X,1)}; % A cell per node containing indices of all data in that node
p = 0; % Vector contiaining the index of the parent node for each node
labels = {}; % A label for each node
key = [];
feature = [];
left_child = [];
right_child = [];

% Create tree by splitting on the root
[inds, p, labels, left_child, right_child, key, feature] = split_node(X, Y, inds, p,labels, cols, ...
    left_child, right_child, key, feature,1);

t.inds = inds;
t.p = p;
t.labels = labels;
t.left_child = left_child;
t.right_child = right_child;
t.key = key;
t.feature = feature;

function [inds, p, labels, left_child, right_child, key, feature] = split_node(X, Y, inds, p, labels, cols,...
    left_child, right_child, key, feature, node)
% Recursively splits nodes based on information gain

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

best_ig = -inf; %best information gain
best_feature = 0; %best feature to split on
best_val = 0; % best value to split the best feature on

curr_X = X(inds{node},:);
curr_Y = Y(inds{node});
% Loop over each feature
for i = 1:size(X,2)
    feat = curr_X(:,i);
    
    % Deterimine the values to split on
    vals = unique(feat);
    if numel(vals) < 2
        continue
    elseif numel(vals) == 2
        split = (vals(1) + vals(2)) / 2;
    else
        split = median(feat);
    end
    
    bin_mat = double(feat < split);
    if (all(bin_mat == 0) || all(bin_mat == 1))
        continue
    end
    H = ent(curr_Y);
    H_cond = cond_ent(curr_Y, bin_mat);
    IG = H - H_cond;
    
    % Find the best split
    if IG > best_ig
        best_ig = IG;
        best_feature = i;
        best_val = split;
    end

end

% Split the current node into two nodes
feat = curr_X(:,best_feature);
%fprintf('best_feature %f\n', best_feature)
feat = feat < best_val;
inds = [inds; inds{node}(feat); inds{node}(~feat)];
inds{node} = [];
p = [p; node; node];
labels = [labels; sprintf('%s < %2.2f', cols{best_feature}, best_val); ...
    sprintf('%s >= %2.2f', cols{best_feature}, best_val)];

n = numel(p)-2;
left_child(node) = n+1;
right_child(node) = n+2;
key(node) = best_val;
feature(node) = best_feature;

% Recurse on newly-create nodes
[inds, p, labels, left_child, right_child, key, feature] = split_node(X, Y, inds, p, labels, cols, ...
    left_child, right_child, key, feature, n+1);
[inds, p, labels, left_child, right_child, key, feature] = split_node(X, Y, inds, p, labels, cols, ...
    left_child, right_child, key, feature, n+2);
