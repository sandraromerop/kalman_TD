
% Make timings as TD learning -- not as R-W learning!
nTrials = 11; 
stimulus = [1 1]; 
% Units:
nbInputU = 10; 
nbHiddenU = 10;

rewards = ones(1,nTrials);
stimuli = repmat(stimulus(:),  1,nbInputU,nTrials  ); 

h  = .001;  % scalar for ant-hebb learning rule
c = .001; % scalar for weight update

% Inputs 
yn = rand(nbHiddenU,1);

B = zeros(nbHiddenU,nbHiddenU);
wn = zeros(nbHiddenU,1); 
% y_0 : to make predictions
% y_inf: to change weights

wnV = zeros(size(wn,1),nTrials);
BV  = zeros(size(B,1),size(B,2),nTrials);
covV= zeros(size(B,1),size(B,2),nTrials);
for it = 1: nTrials 
    rn = rewards(it);
    xn = squeeze(stimuli(:,:,it))';
    xn = xn* eye( length(stimulus)  ,1) ;
    
    if it==1
    y_0 = squeeze(xn) ; % save initial vlaue
    end
    % recurrent dynamics 
    yn =  -yn+xn + B*yn; 
    % Anti hebbian learning rule: 
    y_inf =  inv(eye(size(B,1))-B)*xn ; % shown to be t convergenfce this 
    dB = h*(-xn*yn' + eye(size(B,1))-B);    
    %learning rule for prediction of the weights
    dW = c*(y_inf).*(rn-wn'*y_0); 
    % update:
    B = B + dB; 
    wn = wn + dW;
    
    wnV(: , it) = wn; 
    ynV(:,it)   = yn;
    BV(:,:,it)  = B; 
    covV(:,:,it)= pinv(eye(size(B,1))-B);
    
    
end

 
