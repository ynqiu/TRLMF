function T = trfold(Tk,dims,k,d)
% 
% tensor ring folding method
% yuning.qiu.gd@gmail.com
% 
        N = numel(dims);
%         dims1 = circshift(dims,d-k-1); % k-mode is shift to d-mode
        dims1=circshift(dims,-k+1);
        T = reshape(Tk,dims1);        
        T = shiftdim(T,N-k+1);    % d-mode is shift to k-mode, so T with the original dims.    
end