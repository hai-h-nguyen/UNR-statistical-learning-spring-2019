clear all

%% Load the auto data
M = load('data_processed.txt');

% We want to predict the last column...
Y = M(:,end);
% ...based on the others
X = M(:,2:end-1);

B = 10;

cols = {'sbp', 'tobacco', 'ldl', 'adiposity', 'famhist','typea', 'obesity','alcohol','age'};

n = size(M, 1);
predicted_Y = cell(n,1);

%% Bagging
for i = 1 : B
  %% Bootstrap
  idx_bootstrap = randi([1 n],n,1);

  X_bootstrap = X(idx_bootstrap,:);
  Y_bootstrap = Y(idx_bootstrap,:);
  %% Build the decision tree
  t = build_tree_2(X_bootstrap, Y_bootstrap, cols);
  %% calculate error rate using OOB 
  idx_train = unique(idx_bootstrap);
  
  MX_OOB = M(:, 1:end-1);
  MX_OOB(idx_train, :) = [];
   
  %% calculate major vote for each leaf
  for j = 1 : size(t.inds,1)
    idx_arr = cell2mat(t.inds(j));
    if (~isempty(idx_arr))
        t.major_vote(j) = mode(Y_bootstrap(idx_arr));
    else
        t.major_vote(j) = -1; % invalid
    end
  end
  
  for j = 1 : size(MX_OOB,1)
    %% predict from input
    X_OOB = MX_OOB(j,2:end);
    idx = MX_OOB(j,1);
    node = 1;
    while ((t.left_child(node) > 0) && (t.right_child(node) > 0)) % not a leaf
        if (X_OOB(t.feature(node)) < t.key(node))
            node = t.left_child(node);
        else
            node = t.right_child(node);
        end
    end
    %% compare predicted output and test output
    if (t.major_vote(node) >= 0)
        predicted_Y{idx} = [predicted_Y{idx} t.major_vote(node)];       
    end
  end
  
  %% Display the tree
%   treeplot(t.p');
%   title('Decision tree ("**" is an inconsistent node)');
%   [xs,ys,h,s] = treelayout(t.p');
%   
%   for i = 2:numel(t.p)
%       % Get my coordinate
%       my_x = xs(i);
%       my_y = ys(i);
%       
%       % Get parent coordinate
%       parent_x = xs(t.p(i));
%       parent_y = ys(t.p(i));
%       
%       % Calculate weight coordinate (midpoint)
%       mid_x = (my_x + parent_x)/2;
%       mid_y = (my_y + parent_y)/2;
%       
%       % Edge label
%       text(mid_x,mid_y,t.labels{i-1});
%       
%       % Leaf label
%       if ~isempty(t.inds{i})
%           val = Y(t.inds{i});
%           if numel(unique(val))==1
%               text(my_x, my_y, sprintf('y=%2.2f\nn=%d', val(1), numel(val)));
%           else
%               %inconsistent data
%               text(my_x, my_y, sprintf('**y=%2.2f\nn=%d', mode(val), numel(val)));
%           end
%       end
%   end
end

%% combine predicted value from multiple models
predicted_Y_mode = zeros(n,1);
for i=1:n
    predicted_Y_mode(i) = mode(predicted_Y{i});
end
error_rate_total = sum(predicted_Y_mode ~= Y) / n * 100;
fprintf('Error rate for %d trees:%f %\n', B, error_rate_total);
