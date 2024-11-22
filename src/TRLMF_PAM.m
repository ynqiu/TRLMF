function [X,Y,Out] = TRLMF_PAM(data,omegaIndex,xSize,R,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imbalanced low-rank tensor completion via latent matrix factorization

% Input:
%       data: observed entries of the underlying tensor
%       omegaIndex: indices of observed entries
%       xSize: the dimension of the underlying tensor
%       R: estimated ranks of all mode matricizations
%       opts.
%           maxit: maximum number of iterations (default: 500)
%           tol: stopping tolerance (default: 1e-4)
%           maxT: maximum running time (sec) (default: 1e6)
%           alpha: weights in the model (default: alpha(n) = 1)
%           rho: weights in the proximal term 
%           beta: weights in the Frobenius term
%           alpha: weights in the model (default: alpha(n) = 1)
%           d: the step size of circular unfolding
%
% Output:
%       X,Y: cell structs
%       Out.
%           iter: number of iterations
%           relerr1: relative change array of total fitting
%           relerr2: array of total fitting
%           alpha: final weights alpha (may be different from input)

N = ceil(length(xSize)/2);
Xtemp=zeros(xSize);
Xtemp(omegaIndex)=data;
%% Parameters and defaults
if isfield(opts,'maxit');      maxit = opts.maxit;     else; maxit = 500;             end
if isfield(opts,'tol');        tol = opts.tol;         else; tol = 1e-3;              end
if isfield(opts,'alpha');      alpha = opts.alpha;     else; alpha = ones(1,N);       end
if isfield(opts,'rho');        rho= opts.rho;          else; rho = 1e-1*ones(1,N);    end
if isfield(opts,'beta');        beta= opts.beta;       else; beta = 1e-1*ones(1,N);   end
if isfield(opts,'d');          d= opts.d;              else; d = ceil(length(xSize)/2);end

X = cell(1,N);
Y=cell(1,N);
Xhat=cell(1,N);
Yhat=cell(1,N);
% intialize factor matrices using SVD
for n = 1:N
    [sizeXn,sizeYn]=size(trunfold(1/N*Xtemp,xSize,n,d));
    X{n}=rand(sizeXn,R(n));
    Y{n}=rand(sizeYn,R(n));
    Y{n}=Y{n}';Xhat{n}=X{n};Yhat{n}=Y{n};
end
[omegaIndex,id] = sort(omegaIndex); data = data(id);
 

Mcell=cell(1,N);
for n = 1:N
    Tk=X{n}*Y{n};
    Mn = trfold(Tk,xSize,n,d);
    Mcell{n}=alpha(n)*Mn;
    
end

T=sumCell(Mcell);
T(omegaIndex)=data;
T(~omegaIndex)=mean(data);
Xsq = cell(1,N);  
Ysq=cell(1,N);

for k = 1:maxit
%    fprintf('\b\b\b\b\b%5i',k);
    %% update (X,Y)
    Xk=X; Yk=Y;
    for n = 1: N 
        if alpha(n) > 0
            Msumn=sumCell(Mcell,n);
            Mn=1/alpha(n)*trunfold(T-Msumn,xSize,n,d);
 
                    % update Wn
                    dy=size(Y{n},1);
                    Ysq{n}=Y{n}*Y{n}'+(rho(n)+beta(n))*eye(dy);
                    MYt=rho(n)*Xk{n}+Mn*Y{n}';
                    X{n} = MYt*pinv(Ysq{n});
                    
                    % update Hn
                    Xsq{n} = X{n}'*X{n}+(rho(n)+beta(n))*eye(dy);
                    XtM=X{n}'*Mn+rho(n)*Yk{n};
                    Y{n}=pinv(Xsq{n})*XtM;
                    
                    % collect Wn*Hn
                    Mcell{n}=trfold(alpha(n)*X{n}*Y{n},xSize,n,d);
        end
    end
% update Y
    Tt=T;
    T=(sumCell(Mcell)+1e-3*Tt)/(1+1e-3);
    T(omegaIndex) = data;

    rechs=norm(T(:)-Tt(:))/norm(Tt(:));
    if isfield(opts,'Mtr')
        Out.truerel(k) = norm(T(:)-opts.Mtr(:))/norm(opts.Mtr(:));
        if mod(k, 50)==0
            fprintf('iter=%d, true_rell=%.5f\n',k,Out.truerel(k));
        end
    else 
        Out.rechs(k)=rechs;
        if mod(k, 50)==0
            fprintf('iter=%d, relative change=%.5f\n',k,Out.rechs(k));
        end
    end      


    
    if rechs<tol 
        break;
    end
end 
fprintf('\n'); Out.iter = k;
 
Out.T=T;
Out.X=X;
Out.Y=Y;
Out.Mcell=Mcell;
end

function sumM=sumCell(Mcell,n)
    N=length(Mcell);
    xSize=size(Mcell{1});
    sumM=zeros(xSize);
    if nargin==2
        for i=1:N
            if i~=n
                sumM=sumM+Mcell{i};
            end
        end
    elseif nargin==1
        for i=1:N
            sumM=sumM+Mcell{i};
        end
    end
end

function sumX=sumFro(X)
    N=length(X);
    sumX=0;
    for i=1:N
        Xi=X{i};
        sumX=sumX+norm(Xi(:));
    end
end


