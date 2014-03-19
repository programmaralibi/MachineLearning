[Weights_ji, Weights_kj] = train_nn;
load('Y.mat');

t = zeros(1500,10);


rNum=1;
for i=1:10
    for j = 1:150
        t(rNum,1) = i;
        rNum = rNum+1;
    end
end

X=Y;
N = 1500;
fid = fopen('classes_nn.txt','wt');
activation_hiden = X * Weights_ji';
z_j = tanh(activation_hiden);

M = size(Weights_ji,1);

% Compute z_j_with_bias
z_j(:,2:M+1) = z_j;
z_j(:,1) = 1;

% Compute activation_output
activation_output = z_j * Weights_kj' ;

activation_output = exp(activation_output);
a_ColSum = sum(activation_output, 2);

y_k = activation_output;
classPrediction = zeros(N,10);
misclassificationsPerClass = zeros(10,1);
classLabel = zeros(N,1);
nIncorrectPredictions = 0;
MRR = zeros(10,1);
samplesPerClass = zeros(10,1);


    r_count = 1;
    rec_vector = zeros(1500,1);
    wrprojection = 0;
for i = 1:1500
    [value1 col1] = max(y_k(i,:));
    rec_vector(i,1) = col1; 
    %[value2 col2] = max(t2(i,:));
    fprintf(fid,'%d\n',col1);
    if(col1 ~= t(i))
        wrprojection = wrprojection + 1;
    end
    
        
end
r_ans = zeros(10,1);
for i = 1:10
    r_count = 1;
    j = (i-1)*150 +1;
    while (rec_vector(j)~=i)
        r_count = r_count+1;
        j=j+1;
    end
    
        r_ans(i,1) = r_count;
end
  r_rank = 0;
for i = 1:10
    fprintf('Reciprocal Rank for %d = 1/%d\n',(i-1),r_ans(i));
    r_rank = r_rank + 1/r_ans(i);
end

fprintf('Mean Reciprocal Rank = %d \n',r_rank/10);

fclose(fid);
Error_Rate = (wrprojection/1500);
fprintf('Error Rate = %f %%\n',100*Error_Rate);
