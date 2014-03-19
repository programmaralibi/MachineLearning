function train
format long
load('project1_data.mat');

MLInput = raw_data(:, 2:47);
Target = raw_data(:, 1);

training_array = MLInput(1:6085, :);
training_target = Target(1:6085, 1);

validation_array = MLInput(6086:7607, :);
validation_target = Target(6086:7607, 1);

test_array = MLInput(7607:15211, :);
test_target = MLInput(7607:15211, 1);

ERMS_vector = zeros(5,5,5);
ERMS_vectorTrain = zeros(5,5,5);
ERMS_vectorTest = zeros(5,5,5);

lambda_iterator_vector = zeros(50,50);

degree_val = 1;
rms_lr = 1;
M=1;

for degree=1:5
    mu_val = 1;
    for mu=.1:.1:.5
        [mean_vector, var_vector] = calculate_mean_variance(training_array, mu);
        gaussian_array = create_gaussean_basis_array(training_array, mean_vector, var_vector, degree_val);
        lambda_iterator_val = 1;
        for lambda_iterator=.0001:.0001:.0005
            w_vector = create_w_vector( gaussian_array, training_target, lambda_iterator);
            gaussian_array_validation = create_gaussean_basis_array(validation_array, mean_vector, var_vector, degree_val);
            ERMS_vectorTrain(degree_val, mu_val,lambda_iterator_val) = calculate_error(gaussian_array, w_vector, training_target);
            if(ERMS_vectorTrain(degree_val, mu_val,lambda_iterator_val)< rms_lr)
                rms_lr = ERMS_vectorTrain(degree_val, mu_val,lambda_iterator_val);
                M=degree;
                
            end
            ERMS_vectorTest(degree_val, mu_val,lambda_iterator_val) = calculate_error(gaussian_array_validation, w_vector, validation_target);
            ERMS_vector(degree_val, mu_val,lambda_iterator_val) = calculate_error(gaussian_array_validation, w_vector, validation_target);

            lambda_iterator_vector(degree_val, mu_val, lambda_iterator_val) = lambda_iterator;
            lambda_iterator_val = lambda_iterator_val+1;
        end
        mu_val = mu_val+1;
    end
degree_val = degree_val + 1;
end

gaussian_array_test = create_gaussean_basis_array(test_array, mean_vector, var_vector, 1);

rms_nn = sqrt(nn_model(MLInput, Target));

fprintf('the model complexity M for the linear regression model is %d', M);
fprintf('\n');
fprintf('the regularization parameters lambda_iterator for the linear regression model is %f', lambda_iterator);
fprintf('\n');
fprintf('the root mean square error for the linear regression model is %f', rms_lr);
fprintf('\n');
fprintf('the root mean square error for the neural network model is %f', rms_nn);
fprintf('\n');
end

function [ w_vector] = create_w_vector( gaussian_array,  target_vector, constant )
    lambda_iterator = constant*eye(size(gaussian_array, 2) , size(gaussian_array, 2));
    lambda_iterator(1,1) = 0;
    temp = (gaussian_array'*gaussian_array) + lambda_iterator;
    w_vector = inv(temp)*gaussian_array'*target_vector;
    squared_error = (gaussian_array * w_vector)'*gaussian_array * w_vector;
end

function [complete_array] = create_gaussean_basis_array(array, mean_vector, var_vector, degree)
    for num=1:degree
        basis_array = zeros(size(array));
        for col = 1: size(array,2)
            for row = 1: size(array, 1)
                temp = (array(row, col)-(mean_vector(col)))^2;
                temp1 = (-1/(2*var_vector(col)));
                basis_array(row, col) = exp(temp1*temp);
            end
        end
        additional_col = ones(size(array, 1),1);
        basis_array = [additional_col basis_array];
        if(num==1)
            complete_array = basis_array;
        else
            complete_array = [complete_array basis_array];
        end
    end
end

function [ ERMS ] = calculate_error( gaussian_array, w_vector, target_vector)
    squared_error = (gaussian_array * w_vector - target_vector)'*(gaussian_array * w_vector- target_vector);
    ERMS = sqrt(squared_error / size(gaussian_array, 1));
end

function [ col_mean, col_var ] = calculate_mean_variance( array, degree)
    col_mean = zeros(1,size(array, 2));
    col_var = zeros(1,size(array, 2));
    for i=1:size(array,2)
        col_mean(i) = mean(array(:,i))+(.5*degree);
        col_var(i) = mean(var(array(:,i)));
    end
end