function [score] = evaluate_split(left, right)
	score = 0;

	t_left = left(:, end) - mean(left(:, end));
	t_right = right(:, end) - mean(right(:, end));

	t_left = t_left.^2;    
	score = score + sum(t_left);  


	t_right = t_right.^2;
	score = score + sum(t_right);

end