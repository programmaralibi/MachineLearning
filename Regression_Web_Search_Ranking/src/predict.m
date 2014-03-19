function [ predicted_target ] = predict(gaussian_array, w_vector)
    predicted_target = gaussian_array * w_vector;
end