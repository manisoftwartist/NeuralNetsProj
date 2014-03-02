%  Rosenblatt perceptron for classification
% Last Updated:  31/12/2008
% Update : Modified Regression to Classification

clear all;
 
% Load the training vector set
load  trainvectors1.mat;

% Class labels for classes 1 and 2
Ydes1 =  ones(1,100);
Ydes2 =  -1 * ones(1,100);
Ydes = [ Ydes1 Ydes2];

Y   =   zeros(1,size(U,2));

W = rand(size(U,1),1);  % Assume random weights
eta = 0.1;                      % Learning Rate
misclass_flg = 1;            %  Initialize  a  misclassification  indicator
pass = 0;                       %  1 pass is one iteration through all 2D input points contained in U

while (misclass_flg ==1)
    misclass_flg = 0;
    
    for j = 1 : size(U,2)
        Y(j) = sign( W' * U(:,j) ); % Perceptron Output for each input data point
        if ( Y(j) * Ydes(j) < 0)     % i.e., if Y = -1, but actual class Ydesired = +1,this function would be negative
            misclass_flg = 1;
            W_update = U(:,j) * sign(Ydes(j) - Y(j));
            W = W + W_update;
        end
    end
    pass=pass+1;
    if  (pass==10000)
           misclass_flg = 0;
           disp(' The data points NOT linearly separable...Program exited after 10,000 iterations');
            pass = 0;
    end
end

plot( U(1,1:100), U(2,1:100) , 'b*' );
hold on;
plot( U(1,101:200), U(2,101:end) , 'm+' );


% % Load the testing vector set
load testvectors1.mat; %X is the matrix of test data
mistakes = 200;
 for j = 1 : size(X,2)
        Y(j) = sign( W' * X(:,j) ); % Perceptron Output for each test data point
        if ( Y(j)*Ydes(j) > 0)     % i.e., if Y = -1, but actual class Ydesired = +1,this function would be negative
            mistakes = mistakes-1;
        end
end
disp('No of mistakes');
disp(mistakes);
     
