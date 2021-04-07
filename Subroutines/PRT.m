function y=PRT(m,transp_flag,opts)

% Parabolic Radon Transform operator in frequency domain.
%
% Caculate d=Lm or m=L'd with the transp_flag ('nontransp' or 'transp')
% which L is a function handle.
%
%  Input Parameters:
%       m:---------------------% model vector or data vector in time domain
%       transp_flag:-----------% Caculate d=Lm or m=L'd with the transp_flag
%                                ('nontransp' or 'transp').
% 
%       opts.x_vec:------------% offset vetor ([ min_offset : dx : Max_offset]).
%       opts.q_vec:------------% ray parameter vetor [ q_min : dq : q_max].
%       opts.nt:---------------% the time length of seismic data.
%       opts.nx:---------------% the trace number of seismic data.
%       opts.dt:---------------% the sample interval of seismic data.
%
%  Output Parameters:
%       y:---------------------% the d_adj or m vector in time domain
%
%   Copyright:  Junhai Cao, 13-11-2016.
%   Email:      junhaicao1990@163.com/J.Cao@tudelft.nl
%   Place:      Department of Applied Physics, TU delft
%   Modified:   Junhai Cao, 05-01-2018, TU delft
%               (1) Make the input more simple,and the 'nx,nt,dt,x_vec,q' are in
%                   the "opts" struct;
%  

%% 1. Assign default values to unspecified parameters

% Check for an acceptable number of input arguments
if nargin > 3
    error('**********Error. Too Much Inputs. **********');
end

if nargin < 2
    error('**********Error. Not Enough Inputs. **********');
end

% check dt 
if ~isfield(opts, 'dt')
    error('**********Error. Please input dt **********');
else
    dt=opts.dt;
end

% check nt 
if ~isfield(opts, 'nt')
    error('**********Error. Please input nt **********');
else
    nt=opts.nt;
end

% check nx 
if ~isfield(opts, 'nx')
    error('**********Error. Please input nx **********');
else
    nx=opts.nx;
end

% check x_vec 
if ~isfield(opts, 'x_vec')
    error('**********Error. Please input nx **********');
else
    x_vec=opts.x_vec;
end

% check q_vec vector
if ~isfield(opts, 'q_vec')
    error('**********Error. Please input p_vec **********');
else
    q_vec=opts.q_vec;
end

% check the flow and fhigh
if ~isfield(opts, 'fhigh')
    fhigh=1./(2*dt);
end

if ~isfield(opts, 'flow')
    flow=0;
end


%% 2. Prepare for the main Operator

nq=max(size(q_vec));              % the number of ray parameters p vector
nfft = 2*(2^nextpow2(nt)); % FFT length
df=1/nfft/dt;                 % frequency interval

x_vec=x_vec/max(abs(x_vec));

%---------------------------------------------------
% check the ilow and ihigh (Begin)
%---------------------------------------------------

ilow  = floor(flow*dt*nfft)+1;
if ilow < 2; ilow=2; end;
ihigh = floor(fhigh*dt*nfft)+1;
if ihigh > floor(nfft/2)+1; ihigh=floor(nfft/2)+1; end

%---------------------------------------------------
% check the ilow and ihigh (End)
%---------------------------------------------------

%% 3. The main part for d=Lm or m=L'd

%---------------------------------------------------
% y (Begin)
%---------------------------------------------------
Ifreqnum=ihigh-ilow+1;
% y=zeros(nx*Ifreqnum,1);
% lo=zeros(nx,nq);

% main process
% caculate the L with the frequency lo=exp(i*2*pi*q_vec*x_vec*x_vec)
% and L=lo*exp(f) with f=(ifreq-1)*df.
% for k=1:nx
%     for j=1:nq
%         lo(k,j)=sqrt(-1)*2*pi*q_vec(j)*x_vec(k)*x_vec(k); % Inverse Transform L
%     end
% end

lo = 1i*2*pi*(x_vec.^2)'*q_vec;

if strcmp(transp_flag,'notransp')      % y = A*x_vec
    df_temp=zeros(nx*Ifreqnum,1);
    x_temp=reshape(m,nt,[]);
    m_fp=fft((x_temp),nfft,1);
    m_temp=reshape(m_fp(ilow:ihigh,:)',[],1);
    j=1;
    for ifreq=ilow:ihigh
        L=exp(lo*df*(ifreq-1));
        df_temp(1+(j-1)*nx:nx*j)=L*m_temp(1+(j-1)*nq:nq*j);
        j=j+1;
    end
    
    %             df_temp=L_sparse*m_temp;
    df_transp=reshape(df_temp,nx,[]);
    
    df_new=zeros(nx,nfft);
    df_new(:,ilow:ihigh)=df_transp;
    
    for ii=ilow:ihigh
        df_new(:,nfft+2-ii)=conj(df_new(:,ii));
    end
    df_new(:,nfft/2+1) = zeros(nx,1);
    
    d_time=real(ifft(df_new',[],1));
    d_time=d_time(1:nt,:);
    y=reshape(d_time,[],1);
    
elseif strcmp(transp_flag,'transp') % y = A'*x_vec
    df_temp=zeros(nq*Ifreqnum,1);
    x_temp=reshape(m,nt,[]);
    m_fp=fft(x_temp,nfft,1);
    m_temp=reshape(m_fp(ilow:ihigh,:)',[],1);
    
    j=1;
    for ifreq=ilow:ihigh
        L=exp(lo*df*(ifreq-1));
        df_temp(1+(j-1)*nq:nq*j)=L'*m_temp(1+(j-1)*nx:nx*j);
        j=j+1;
    end
    
    %             df_temp=L_sparse'*m_temp;
    df_transp=reshape(df_temp,nq,[]);
    
    m_new=zeros(nq,nfft);
    m_new(:,ilow:ihigh)=df_transp;
    
    for ii=ilow:ihigh
        m_new(:,nfft+2-ii)=conj(m_new(:,ii));
    end
    m_new(:,nfft/2+1) = zeros(nq,1);
    
    tp=real(ifft(m_new',[],1));
    %             tp=tp(1:nt,:);
    tp=tp(1:nt,:);
    y=reshape(tp,[],1);
    
end

%---------------------------------------------------
% y  (End)
%---------------------------------------------------

end



