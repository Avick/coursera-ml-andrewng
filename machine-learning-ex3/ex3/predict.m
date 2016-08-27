function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%oldXsize = size(X)
X = [ones(m, 1) X];
%newXsize = size(X)

%Theta1size = size(Theta1)
%Theta2size = size(Theta2)

second_layer_matrix = sigmoid(X*transpose(Theta1));
%for (i = 1: size(Theta1,1) )
%  second_layer_row = sigmoid(X * Theta1(i,:)(:));
%  second_layer_matrix = [second_layer_matrix, second_layer_row];
%end

second_layer_matrix = [ones(size(second_layer_matrix,1), 1) second_layer_matrix];
%temp = size(second_layer_matrix)
output_layer_matrix = [];
for (i = 1: num_labels )
  output_layer_row = sigmoid(second_layer_matrix * Theta2(i,:)(:));
  output_layer_matrix = [output_layer_matrix, output_layer_row];
end

max_row_element_matrix = max(output_layer_matrix, [], 2);

for(i = 1: m)
  p(i) = find(output_layer_matrix(i,:) == max_row_element_matrix(i));
end

% =========================================================================


end
