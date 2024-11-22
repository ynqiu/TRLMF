function  Tk = trunfold(T,dims,k,d)
% 
% tensor ring unfolding method
% yuning.qiu.gd@gmail.com
% 
%      N = numel(dims);
     T = shiftdim(T,k-1);
     % T = shiftdim(T,N+k-d+1);%+1     % k-mode is shift to d-mode
     dims1 = size(T); % k-mode is shift to d-mode
     Tk = reshape(T,prod(dims1(1:d)),[]);    
end