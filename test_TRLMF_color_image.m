clear; close all;
addpath('src')
image_dir = 'images';
image_name = 'lena.bmp';
Xim = double(imread([image_dir,'/',image_name]));

sr = 0.2;
xSize = size(Xim);
omegaIndex = randperm(prod(xSize),round(sr*prod(xSize)));
Xo = zeros(xSize);
Xo(omegaIndex) = Xim(omegaIndex);
Omega = zeros(xSize);
Omega(omegaIndex) = 1;
N=length(xSize);

% tensorize the data
Ximh = reshape(Xim,[4,4,4,4, 4,4,4,4, 3]);

Xoh = reshape(Xo,[4,4,4,4, 4,4,4,4, 3]);
Omegah = reshape(Omega,[4,4,4,4, 4,4,4,4, 3]);
omegaIndexh = find(Omegah == 1);

xSizeh = size(Xoh);
Nh = length(xSizeh);

% set some paramters for TRLMF
r = 20;
alpha = ones(1,Nh);
rho = 0.5*ones(1,Nh);
beta = 100*ones(1,Nh);

data = Ximh(omegaIndexh);
R=r*ones(1,Nh);
optsRun=struct('maxit',800,'tol',1e-4,'rho',rho,'alpha',alpha,'beta',beta);

[W,H,Out] = TRLMF_PAM(data,omegaIndexh,xSizeh,R,optsRun);
costTime=toc;
Xhath=Out.T;
Xhat = reshape(Xhath, [256,256,3]);
RSE = norm(Xim(:)-Xhat(:)) / norm(Xim(:));
fprintf('RSE=%.4f',RSE);

% plot the results
figure;
subplot(1,3,1);
imshow(uint8(Xim));
title("Original");
subplot(1,3,2);
imshow(uint8(Xo));
title_name_ob = ["Observed, SR=",num2str(sr)];
title(title_name_ob);
subplot(1,3,3);
imshow(uint8(Xhat));
title_name = ["Recoverd, RSE=",num2str(RSE)];
title(title_name)