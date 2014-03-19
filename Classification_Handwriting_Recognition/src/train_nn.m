function [Weights_ji, Weights_kj] = train_nn
clear all;
load('X.mat');
row_num_=zeros(11,1);
row_num_(2)=2000;
row_num_(3)=1979;
row_num_(4)=1999;
row_num_(5)=2000;
row_num_(6)=2000;
row_num_(7)=2000;
row_num_(8)=2000;
row_num_(9)=2000;
row_num_(10)=2000;
row_num_(11)=2000;
N = size(X,1);
D = 513;        % 512 dimensions + bias
K = 10;
t=zeros(1500,10);
r=1;
r1=1;
for i=1:10
    r=row_num_(i)+r;
    r1=row_num_(i+1)+r1;
    for j=r:(r1-1)
        t(j,i)= 1;
    end
end

%%
M = 20;
eta = 0.00003;
maxEp = 1000;
Weights_ji = rand(M, D) - 0.5;
Weights_kj = rand(K, M + 1) - 0.5;

output_hodden_with_bias = ones(N,M+1);
times=0;
o_E = inf;
oldWkj = Weights_kj;
oldWji = Weights_ji;

nCrossedMinima = 0;
nIterSince = 0;
nMaxIterations=150;
EVer = zeros(nMaxIterations,1);
EtaVer = zeros(nMaxIterations,1);
delEkjVsIter = zeros(nMaxIterations,1);
delEjiVsIter = zeros(nMaxIterations,1);

for epoch = 1:maxEp
    
    activation_hodden = X * Weights_ji';
    output_hodden = tanh(activation_hodden);
    output_hodden_with_bias(:,2:end) = output_hodden;
    output_hodden_with_bias(:,1) = 1;
    a_k = output_hodden_with_bias * Weights_kj' ;
    a_k = exp(a_k);
    a_ColSum = sum(a_k, 2);

    yk = a_k;
    for i = 1:N
        yk(i,:) = yk(i,:) / a_ColSum(i);
    end
    E = -1 * sum(sum(t .* log(yk)));
    EVer(epoch) = E;
    
    nIterSince = nIterSince + 1;
    if (o_E > E)
        o_E = E;
        oldWkj = Weights_kj;
        oldWji = Weights_ji;
    end
    if (o_E < E)
        if nIterSince <= 10
            break;
        end
        
        nCrossedMinima = nCrossedMinima + 1;
        nIterSince = 0;
        eta = eta / (epoch ^ (1/3));
        Weights_kj = oldWkj;
        Weights_ji = oldWji;

        continue;
    end
    
    % Compute del_k
    del_k = yk - t;
    
    % Compute dhoE_by_dhoWkj
    dhoE_by_dhoWkj = del_k' * output_hodden_with_bias;
    
    delEkjVsIter(epoch) =  sum(dhoE_by_dhoWkj(:));

    % Compute SUM(del_k * Weights_kj) over k
    del_j_with_bias = del_k * Weights_kj;
    del_j = del_j_with_bias(:,2:end);

    % Compute dhoE_by_dhoWji
    dhoE_by_dhoWji = (del_j .* (ones(N,M) - output_hodden.^2))' * X;
    
    delEjiVsIter(epoch) =  sum(dhoE_by_dhoWji(:));
    % Update output layer weights
    Weights_kj = Weights_kj - eta * dhoE_by_dhoWkj;

    % Update hidden layer weights
    Weights_ji = Weights_ji - eta * dhoE_by_dhoWji;
    
    %disp(['########   Network error at epoch #', num2str(epoch), ' = ', num2str(E)]);
    error(epoch)=E;
    eta = eta + epoch*(100/E^3)*(1/(nCrossedMinima + 1));
    EtaVsIter(epoch) = eta;
    times=epoch;
    
end

% Verify training set classification with generated model
activation_hodden = X * Weights_ji';
output_hodden = tanh(activation_hodden);

% Compute output_hodden_with_bias
output_hodden_with_bias(:,2:end) = output_hodden;
output_hodden_with_bias(:,1) = 1;

% Compute a_k
a_k = output_hodden_with_bias * Weights_kj' ;

a_k = exp(a_k);
a_ColSum = sum(a_k, 2);

yk = a_k;
classPrediction = zeros(N,10);
misclassificaPerClass = zeros(10,1);
nIncorrectPredictions = 0;
MRR = zeros(10,1);
samplesPerClass = zeros(10,1);
for i = 1:N
    yk(i,:) = yk(i,:) / a_ColSum(i);
    
    % Find the class which is deemed to be most probable
    [maxY, k] = max(yk(i,:));

    % Set that class to 1 as prediction result
    classPrediction(i,k) = 1;

    % Check if the classPrediction is same as target
    bEqual = isequal(classPrediction(i,:),t(i,:));

    % Find the actual class this sample belongs to - t_k
    [maxT, t_k] = max(t(i,:));
    
    
    if (bEqual ~= true)
        nIncorrectPredictions = nIncorrectPredictions + 1;
        misclassificaPerClass(t_k) = misclassificaPerClass(t_k) + 1;
    end
    
    prob_y_k = yk(i,t_k);
    t_y_k = yk(i,:);
    t_y_k = sort(t_y_k, 'descend');
    
    rank = find(t_y_k == prob_y_k);
    
    reciprocal_rank = 1 / rank;
    
    MRR(t_k) = MRR(t_k) + reciprocal_rank;
    samplesPerClass(t_k) = samplesPerClass(t_k) + 1;
    
end
plot (1:times,error);
xlabel('Varying iteration ');
ylabel('Error');

nClassificationRate = ((N-nIncorrectPredictions) / N) * 100;
disp([num2str(nClassificationRate), '% correct classification of training data after ', num2str(epoch), ' epochs ********** ']);

MRR = MRR ./ samplesPerClass;

end