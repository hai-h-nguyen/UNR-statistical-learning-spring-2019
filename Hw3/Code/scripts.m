
clc;
clear;

data_table = importdata('zip.train');

cluster_0 = [];
cluster_1 = [];
cluster_2 = [];
cluster_3 = [];
cluster_4 = [];
cluster_5 = [];
cluster_6 = [];
cluster_7 = [];
cluster_8 = [];
cluster_9 = [];



for i = 1:length(data_table)
    cluster_id = data_table(i,1); 
    
    if (cluster_id == 0)
    	if (isempty(cluster_0))
    		cluster_0 = data_table(i, 2:end);
    	else
    		cluster_0 = vertcat (cluster_0, data_table(i, 2:end));
    	end
    end

    if (cluster_id == 1)
    	if (isempty(cluster_1))
    		cluster_1 = data_table(i, 2:end);
    	else
    		cluster_1 = vertcat (cluster_1, data_table(i, 2:end));
    	end
    end

    if (cluster_id == 2)
    	if (isempty(cluster_0))
    		cluster_2 = data_table(i, 2:end);
    	else
    		cluster_2 = vertcat (cluster_2, data_table(i, 2:end));
    	end
    end

    if (cluster_id == 3)
    	if (isempty(cluster_0))
    		cluster_3 = data_table(i, 2:end);
    	else
    		cluster_3 = vertcat (cluster_3, data_table(i, 2:end));
    	end
    end

     if (cluster_id == 4)
    	if (isempty(cluster_0))
    		cluster_4 = data_table(i, 2:end);
    	else
    		cluster_4 = vertcat (cluster_4, data_table(i, 2:end));
    	end
    end           

    if (cluster_id == 5)
    	if (isempty(cluster_0))
    		cluster_5 = data_table(i, 2:end);
    	else
    		cluster_5 = vertcat (cluster_5, data_table(i, 2:end));
    	end
    end

    if (cluster_id == 6)
    	if (isempty(cluster_0))
    		cluster_6 = data_table(i, 2:end);
    	else
    		cluster_6 = vertcat (cluster_6, data_table(i, 2:end));
    	end
    end

    if (cluster_id == 7)
    	if (isempty(cluster_0))
    		cluster_7 = data_table(i, 2:end);
    	else
    		cluster_7= vertcat (cluster_7, data_table(i, 2:end));
    	end
    end

    if (cluster_id == 8)
    	if (isempty(cluster_0))
    		cluster_8 = data_table(i, 2:end);
    	else
    		cluster_8 = vertcat (cluster_8, data_table(i, 2:end));
    	end
    end

     if (cluster_id == 9)
    	if (isempty(cluster_0))
    		cluster_9 = data_table(i, 2:end);
    	else
    		cluster_9 = vertcat (cluster_9, data_table(i, 2:end));
    	end
    end     
end



