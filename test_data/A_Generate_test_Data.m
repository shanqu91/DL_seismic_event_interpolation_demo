%  Generate Radon panel and Parabolic siemic data 
%
%      (1): generate Radon panel
%      (2): Radon domain--->seismic data
%      (3): save figures and data
%
%   Copyright:  Junhai Cao, 02-03-2017.
%   Email:      junhaicao1990@163.com/J.Cao@tudelft.nl
%   Place:      Department of Applied Physics, TU delft

clc;clear; close all;
%% 1. add path of software package or Matlab function
addpath('../Subroutines')

%% 2. set values to the parameters 

dq=0.005;           % slowness interval
qmin=0;             % min slowness 
qmax=0.5;           % max slowness

q_vec=qmin:dq:qmax; %ray parameter (q) range
nq=size(q_vec,2);   % the number of q_vec

dx=5;               % trace interval
xmin=0;             % min offset
xmax=300;           % max offset
x_vec=xmin:dx:xmax; % offset axis
nx=size(x_vec,2);   % trace number

% the tau and q_vec values of seismic event
tau = [0.003, 0.1, 0.2, 0.28]; 
q_sys = [0.45, 0.25, 0.2, 0.12];

dt=2.0/1000;        % sample interval
nt=201;            % time length of data
t_vec=0:dt:(nt-1)*dt; % time axis

% generate ricker wavelet 
freq=30;            % frequency of the wavelet
[w,t]=rickerw(freq,dt); % make ricker wavelet
W_length=size(w,1);     % ricker wavelet length

% analysis the aliasing conditions
x2_max=max(x_vec.^2);
x2_min=min(x_vec.^2);

if dq<=(1/freq/(x2_max-x2_min)*x2_max) && dx<=(x2_max/freq/xmax/qmax)
    %fprintf('----------------------------------------\n');
    %disp('antialising conditions are satisfied');
    %fprintf('----------------------------------------\n');
else 
    error('antialising conditions are not satisfied, please try again');
end

clear x2_max x2_min;
%% 3. generate the model(Tau-q_vec domain)
m=zeros(nt,nq);

for i=1:size(tau,2)
    
    temp_t=round(tau(i)/dt)+1;
    if q_sys(i)>=0
        temp_p=round(q_sys(i)/dq)+1+round(abs(qmin)/dq);
    else
        temp_p=round(abs(q_sys(i))/dq)+1;
    end
    
    m(temp_t:W_length+temp_t-1,temp_p)=w;
    
end

clear temp_t temp_p;
%% 4. model---> seismic data(x_vec-t domain)
opts.dt=dt;
opts.nt=nt;
opts.x_vec=x_vec;
opts.nx=max(size(x_vec));
opts.q_vec=q_vec;

d=PRT(reshape(m,[],1),'notransp',opts); % d= Lm
m_adj=PRT(reshape(d,[],1),'transp',opts); % m = L'*d

%% 5. seismic data---> tau-q_vec domain

%% 6. save Model and seismic data
orig_data = reshape(d,nt,nx);

save('../Data/orig_data_test.mat', 'orig_data', '-v7')

%% disappear traces randomly, 50% traces are left
load ../Data/orig_data_test.mat
percent = 50;       % percentage traces left
X_test = select_traces(orig_data, 50);

figure;imagesc([X_test orig_data])
title('test data (left-input, right-ideal output)')
print -djpg ../Fig/X_test.jpg

save('../Data/X_test.mat', 'X_test', '-v7')

waitfor(gcf)
