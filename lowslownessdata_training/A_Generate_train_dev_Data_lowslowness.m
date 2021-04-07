%  Generate Radon panel and Parabolic siemic train and dev data with slowness range: 0-0.5
%
%      (1): generate Radon panel
%      (2): Radon domain--->seismic data
%      (3): save figures and data
%
%   Copyright:  Modified by Shan Qu, originallly written by Junhai Cao, 02-03-2017.
%   Email:      S.Qu@tudelft.nl/junhaicao1990@163.com/J.Cao@tudelft.nl
%   Place:      Department of Applied Physics, TU delft

clc;clear; close all;
addpath('../Subroutines')

%% 2. generate training data with slowness range: 0-0.25

dq=0.005;           % slowness interval
qmin=0.0;           % min slowness 
qmax=0.2;           % max slowness

q_vec=qmin:dq:qmax; %ray parameter (q) range
nq=size(q_vec,2);   % the number of q_vec

dx=5;               % trace interval
xmin=0;             % min offset
xmax=300;           % max offset
x_vec=xmin:dx:xmax; % offset axis
nx=size(x_vec,2);   % trace number

N1 = 400;
N2 = 4;
N12 = N1 * N2;
tau_set = cell(N12, 1);
q_sys_set = cell(N12, 1);
tau_base = zeros(9, N1);
q_sys_base = zeros(9, N1);
for j = 1 : N1
    a = 0.003;
    b = 0.3;
    for k = 1 : 9
        tau_base(k, :) = (b-a).*rand(N1,1) + a;
    end
    a = qmin;
    b = qmax;
    for k = 1 : 9
        q_sys_base(k, :) = (b-a).*rand(N1,1) + a;
    end

    for i = 1 : N2
        tau_set{i+(j-1)*N2} = tau_base(1:i+1, j)';
        q_sys_set{i+(j-1)*N2} = q_sys_base(1:i+1, j)';
    end
end

freq_set = 28:32;
N_freq = size(freq_set, 2);
d_set = cell(N_freq*N12, 1);


for n_freq = 1 : N_freq
    for n12 = 1 : N12
        disp(['#' num2str((n_freq-1)*N12 + n12) 'data']);
        % the tau and q_vec values of seismic event
        tau = tau_set{n12}; 
        q_sys = q_sys_set{n12};

        dt=2.0/1000;        % sample interval
        nt=201;            % time length of data
        t_vec=0:dt:(nt-1)*dt; % time axis

        % generate ricker wavelet 
        freq=freq_set(n_freq);            % frequency of the wavelet
        [w,t]=rickerw(freq,dt); % make ricker wavelet
        W_length=size(w,1);     % ricker wavelet length

        % analysis the aliasing conditions
        x2_max=max(x_vec.^2);
        x2_min=min(x_vec.^2);

        if dq<=(1/freq/(x2_max-x2_min)*x2_max) && dx<=(x2_max/freq/xmax/qmax)
           % fprintf('----------------------------------------\n');
           % disp('antialising conditions are satisfied');
           % fprintf('----------------------------------------\n');
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

        %% 6. save Model and seismic data
        d_set{(n_freq-1)*N12 + n12} = reshape(d,nt,nx);
        if (mod((n_freq-1)*N12 + n12 - 1, 100) == 0)
            figure(1);imagesc(d_set{(n_freq-1)*N12 + n12})
            title(['original data ' num2str((n_freq-1)*N12 + n12) ' for train and dev'])
        end
    end
end

orig_data = zeros(N_freq*N12, 201, 61);
for i = 1 : N_freq*N12
    orig_data(i, :, :) = d_set{i};
end


orig_data = shuffle( orig_data , 1 );
orig_data = shuffle( orig_data , 1 );
orig_data = shuffle( orig_data , 1 );

save('../Data/orig_data_lowslowness.mat', 'orig_data', '-v7')

fprintf('----------------------------------------\n');
disp('train and development original data are generated');
fprintf('----------------------------------------\n');

%% disappear traces randomly, 50% traces are left
N1 = 400;
N2 = 4;
N12 = N1 * N2;

freq_set = 28:32;
N_freq = size(freq_set, 2);
N = N_freq*N12;

%% train and dev data with slowness range: 0-0.25
load ../Data/orig_data_lowslowness.mat

percent = 50;       % percentage traces left
missing_data = zeros(size(orig_data));
for i = 1 : N
    missing_data(i, :, :) = select_traces(squeeze(orig_data(i, :, :)), percent);
end

save('../Data/missing_data_lowslowness.mat', 'missing_data', '-v7')

X_train = missing_data(1:N*0.8, :, :);
X_dev = missing_data(N*0.8+1:end, :, :);

Y_train = orig_data(1:N*0.8, :, :);
Y_dev = orig_data(N*0.8+1:end, :, :);

save('../Data/X_train_lowslowness.mat', 'X_train', '-v7')
save('../Data/X_dev_lowslowness.mat', 'X_dev', '-v7')
save('../Data/Y_train_lowslowness.mat', 'Y_train', '-v7')
save('../Data/Y_dev_lowslowness.mat', 'Y_dev', '-v7')

fprintf('----------------------------------------\n');
disp('train (80%) and development (20%) data are generated (50% traces are missing randomly)');
fprintf('----------------------------------------\n');

figure;imagesc([squeeze(X_train(100,:,:)) squeeze(Y_train(100,:,:))])
title('train data example 0 (left-input, right-output)')
print -djpg ../Fig/traindata_lowslowness_0.jpg
figure;imagesc([squeeze(X_train(500,:,:)) squeeze(Y_train(500,:,:))])
title('train data example 1 (left-input, right-output)')
print -djpg ../Fig/traindata_lowslowness_1.jpg
figure;imagesc([squeeze(X_train(700,:,:)) squeeze(Y_train(700,:,:))])
title('train data example 2 (left-input, right-output)')
print -djpg ../Fig/traindata_lowslowness_2.jpg

waitfor(gcf)
