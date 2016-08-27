  function [J grad] = nnCostFunction(nn_params, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, ...
                                     X, y, lambda)
  %NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices. 
  % 
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.
  %
  
  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));
  
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
  
  % Setup some useful variables
  m = size(X, 1);
       
  % You need to return the following variables correctly 
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the code by working through the
  %               following parts.
  %
  % Part 1: Feedforward the neural network and return the cost in the
  %         variable J. After implementing Part 1, you can verify that your
  %         cost function computation is correct by verifying the cost
  %         computed in ex4.m
  %
  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  %
  %         Note: The vector y passed into the function is a vector of labels
  %               containing values from 1..K. You need to map this vector into a 
  %               binary vector of 1's and 0's to be used with the neural network
  %               cost function.
  %
  %         Hint: We recommend implementing backpropagation using a for-loop
  %               over the training examples if you are implementing it for the 
  %               first time.
  %
  % Part 3: Implement regularization with the cost function and gradients.
  %
  %         Hint: You can implement this around the code for
  %               backpropagation. That is, you can compute the gradients for
  %               the regularization separately and then add them to Theta1_grad
  %               and Theta2_grad from Part 2.
  %
  
  %temp = input_layer_size
  %temp1 = hidden_layer_size
  X = [ones(m, 1) X];
  HiddenLayerMatrix = sigmoid(X*transpose(Theta1));
  HiddenLayerMatrix = [ones(size(HiddenLayerMatrix,1), 1) HiddenLayerMatrix];
  OutputLayerMatrix = sigmoid(HiddenLayerMatrix*transpose(Theta2));
  
  labels = 1:num_labels;
  delta3 = zeros(num_labels,1);
  delta2 = zeros(hidden_layer_size,1);
  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));
  %temp = size(Delta2)
  for i = 1: m
    c = y(i);
    pos = find(labels == c);
    neg = find(labels != c);
    
    %For Part 1
    J += (sum(-log(OutputLayerMatrix(i,pos))) + sum(-log(1-OutputLayerMatrix(i,neg))))/m;
    
          %For Part 2
    delta3(pos) = OutputLayerMatrix(i,pos) - 1;
    delta3(neg) = OutputLayerMatrix(i,neg);
    
    delta2 = ((transpose(Theta2)*delta3).*(HiddenLayerMatrix(i,:)(:)).*(1 - HiddenLayerMatrix(i,:)(:)))(2:end);  
    
    %∆ (l) = ∆ (l) + δ (l+1) (a (l) ) T
    Delta1 +=  delta2*X(i,:);
    Delta2 +=  delta3*HiddenLayerMatrix(i,:);  
    
  end
  
    
%    output=y;
%    y=zeros(num_labels,m);
%  
%  for i=1:size(output,1)
%    y(output(i),i)=1;
%  end
%    DELTA11=zeros(size(Theta1,1),size(Theta1,2));
%    DELTA21=zeros(size(Theta2,1),size(Theta2,2));
  
%   for t=1:m
%    a1=X(t,:);
%    a1=transpose(a1);
%    z2=Theta1*a1;
%    a2=sigmoid(z2);
%    a2=[1;a2];
%    z3=Theta2*a2;
%    a3=sigmoid(z3);
%    Ddelta3=a3 - y(:,t);
%    Ddelta2=(transpose(Theta2)*Ddelta3).*(a2.*(1-a2));
%    Ddelta2=Ddelta2(2:end);
%    DELTA21=DELTA21+(delta3*transpose(a2));
%    DELTA11=DELTA11+(delta2*transpose(a1));
%  
%  end
        %temp = size(OutputLayerMatrix(m,:))
        %temp = size(a3)
        %temp = [OutputLayerMatrix(m,:)(:), a3]
        %y_dash = zeros(10,1);
        %y_dash(pos) = 1;
        %temp = [y_dash , y(:,m)]
        %temp = [OutputLayerMatrix(m)(:), z3]
        %temp = [delta3,Ddelta3]
        %sigmoidGradient(HiddenLayerMatrix(m))
        %temp = [HiddenLayerMatrix(m,:)(:),a2]
        %temp = [(HiddenLayerMatrix(i,:)(:)).*(1 - HiddenLayerMatrix(i,:)(:)), a2.*(1-a2)]
        %temp = [transpose(sigmoidGradient(X*transpose(Theta1)))(:,m), a2.*(1-a2)]
        %temp = [delta2,Ddelta2]
        %temp = [HiddenLayerMatrix(m,:)(:) ,a2]
        %temp = [Delta1(:,1), DELTA11(:,1)]
  
  J += lambda*(sum(Theta1(hidden_layer_size+1:end).^2) + sum(Theta2(num_labels+1:end).^2))/(2*m);
  
  % -------------------------------------------------------------
  % Regularization
  
  Theta1_grad = Delta1/m;
  Theta2_grad = Delta2/m;
  Theta1_grad( : , 2:end ) += (lambda/m)*Theta1(:,2:end);
  Theta2_grad( : , 2:end ) += (lambda/m)*Theta2(:,2:end);
  % =========================================================================
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
  
  end
