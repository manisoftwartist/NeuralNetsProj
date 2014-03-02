%  WORKING - Do not modify
% Rosenblatt perceptron with less number of iterations

clear all;

load trainvectors2.mat;

W = [0.1;0.3]; 
noise = randn( [1 (size(U,2))] );
% Assumed underlying function is y = f(u)+noise; - Linear Regression?
Ydes =  (W'*U)+noise;

% Network Outputs pre-allocated
Y      =   zeros( [1 size(U,2)]);

echo on;
more off;

imax = 500;  
max_iterations = (imax*1000);

% W_chkpts = ones([2 (max_iterations/imax)]);
W_chkpts = ones([2 10]);

p = 1; % Array index for weight matrix at instants from i to i max
for i = 1: max_iterations
        if (mod(i,10) ~=0)                     % Check after (10*i) iterations
            chk_flg = 0;
        else
            chk_flg = 1;
        end
        
        M = size(U,2);                                              % No of incorrect classifications in each iteration of the below for loop
        for col = 1: size(U,2)
           Y(col) = W' * U( : , col);
           if ( sign(Y(col)) == Ydes(col))
               M = M - 1;
           end
            if (sign(Y(col)) ~= Ydes(col))                              % If the image is not classified correctly 
                    if (chk_flg ==1)
                           W_chkpts(1,p)   =    W(1,:);    % Weight after every 1000 iterations; at the instants (1000*i)
                           W_chkpts(2,p)   =    W(2,:);
                   end
                    wt_update = U( : , col) * (sign(Ydes(col) - Y(col)));
                    W = W + wt_update;
            end
        end
        
        % Finish training if all images are classified correctly;else
        % continue till max_iterations
        if (chk_flg ==1)
             p = p+1;
              if (M ==0)
                 disp(' Images are classified correctly before the max no if iterations'); 
                 break;
              end
        end
end % outermost for - learning iterations

t = 1 : imax;
plot( t , W_chkpts(1,:) ,'r');
hold on;
plot( t , W_chkpts(2,:) );
