classdef mdpMain < handle
% mdpMain Compute the optimal multi-modality cancer treatment policy using
% backward induction for stationary transition probabilities.

    properties
       n          % number of state variables
       m          % number of states per state variable
       numStates  % total number of states
       numActions % number of actions
       N          % number of epochs
       r          % reward functions
       Pn         % side-effect and tumor transition probabilities
       P          % transition probability matrix
       A          % optimal policy
       tol        % tie threshold
    end
    
    methods
        function mdp = mdpMain(pars)
        % Initialize mdp variables according to pars and/or default values.
        
            flag = exist('pars','var');
            mdp.n = 3;
            if flag && isfield(pars,'m')
                mdp.m = pars.m;
            else
                mdp.m = [2,10,10];
            end
            mdp.numStates = prod(mdp.m);
            if flag && isfield(pars,'numActions')
                mdp.numActions = pars.numActions;
            else
                mdp.numActions = 3;
            end
            if flag && isfield(pars,'N')
                mdp.N = pars.N;
            else
                mdp.N = 3;
            end
            if flag && isfield(pars,'r')
                mdp.r = pars.r;
            else
                mdp.r{1} = @(S)0;
                mdp.r{2} = @(S)100*((1/3)*(mdp.m(2)^2-S(2)^2)/mdp.m(2)^2+...
                    (2/3)*(mdp.m(3)^2-S(3)^2)/mdp.m(3)^2);
            end
            if flag && isfield(pars,'Pn')
                mdp.Pn = pars.Pn;
            else
                mdp.Pn{1,1} = [0 40 60]; % M1 side-effect
                mdp.Pn{1,2} = [0 60 40]; % M2 side-effect
                mdp.Pn{1,3} = [60 40 0]; % M3 side-effect
                mdp.Pn{2,1} = [70 30 0]; % M1 tumor
                mdp.Pn{2,2} = [60 40 0]; % M2 tumor
                mdp.Pn{2,3} = [0 30 70]; % M3 tumor
            end
            mdp.tol = 1e-6;
            
            % check transition probability assumptions
            mdp.checkAssumptions();
        end
        
        function flag = checkAssumptions(mdp)
        % Check transition probability assumptions.
        
            flag = 0;
        
            % no negative probability values
            for i = 1:mdp.numActions
                sPn = mdp.Pn{1,i};
                tPn = mdp.Pn{2,i};
                for j = 1:mdp.n
                    if sPn(j) < 0 || tPn(j) < 0
                        flag = 1;
                        disp('Assumption violation: Negative transition probability.')
                    end
                end
            end
        
            % surveillance transition probabilities
            sPn_surveillance = mdp.Pn{1,mdp.numActions};
            tPn_surveillance = mdp.Pn{2,mdp.numActions};
            
            % side-effect does not increase with surveillance
            if sPn_surveillance(3) > 0
                flag = 1;
                disp('Assumption violation: Increase in side-effect due to surveillance.')
            end
            
            % tumor does not decrease with surveillance
            if tPn_surveillance(1) > 0
                flag = 1;
                disp('Assumption violation: Decrease in tumor due to surveillance.')
            end
            
            for i = 1:mdp.numActions-1
                sPn_treatment = mdp.Pn{1,i};
                tPn_treatment = mdp.Pn{2,i};
                
                % tumor does not increase with treatment
                if tPn_treatment(3) > 0
                    flag = 1;
                    disp('Assumption violation: Increase in tumor due to treatment.')
                end
                
                % side-effect does not decrease with treatment
                if sPn_treatment(1) > 0
                    flag = 1;
                    disp('Assumption violation: Decrease in side-effect due to treatment.')
                end
                
                % treatment is better than no treatment
                if sPn_surveillance(1) > tPn_treatment(1)
                    flag = 1;
                    disp('Assumption violation: Decrease in side-effect due to surveillance > decrease in tumor due to treatment.')
                end
                if tPn_surveillance(3) < sPn_treatment(3)
                    flag = 1;
                    disp('Assumption violation: Increase in tumor due to surveillance < increase in side-effect due to treatment.')
                end
            end
            
            sPn_M1 = mdp.Pn{1,1};
            tPn_M1 = mdp.Pn{2,1};
            for i = 2:mdp.numActions-1
                sPn_treatment = mdp.Pn{1,i};
                tPn_treatment = mdp.Pn{2,i};
                
                % M1 has highest risk (side-effect increase)
                if sPn_M1(3) < sPn_treatment(3)
                    flag = 1;
                    disp('Assumption violation: M1 does not have highest side-effect increase.')
                end
            
                % M1 has highest reward (tumor decrease)
                if tPn_M1(1) < tPn_treatment(1)
                    flag = 1;
                    disp('Assumption violation: M1 does not have highest tumor decrease.')
                end
            end
        end
        
        function S = calcStates(mdp)
        % Create the matrix of all possible states.  
        
            S = zeros(mdp.numStates,mdp.n);
            for i = 1:mdp.numStates
                temp1 = i-1;
                for j = mdp.n:-1:1
                    temp2 = mod(temp1,prod(mdp.m(j:end)));
                    S(i,j) = temp2/prod(mdp.m(j+1:end))+1;
                    temp1 = temp1 - temp2;
                end
            end
            S = S - 1;
        end
        
        function R = calcReward(mdp,t,S)
        % Calculate the immediate or terminal reward vector.
        
            rt = mdp.r{t};
            R = zeros(mdp.numStates,1);
            for i = 1:mdp.numStates
                R(i) = rt(S(i,:));
            end
        end
        
        function calcProb(mdp)
        % Calcuate the transition probability matrix.

            mdp.P = zeros(mdp.numStates,mdp.numStates,mdp.numActions);
            for i = 1:mdp.numActions
                % extract and normalize
                sProbVec = mdp.Pn{1,i}/sum(mdp.Pn{1,i});
                tProbVec = mdp.Pn{2,i}/sum(mdp.Pn{2,i});

                % create OAR and tumor transition probability matrices
                oarProbMat = mdp.calcProbMat(mdp.m(2),sProbVec,'o');
                tProbMat = mdp.calcProbMat(mdp.m(3),tProbVec,'t');
                
                % combine OAR and tumor matrices with absorbing states
                P0 = kron(oarProbMat,tProbMat);
                idx = (mdp.m(2) - 1)*mdp.m(3) + 1;
                P0(idx:end,idx:end) = eye(mdp.m(3)); % OAR death
                for j = mdp.m(3):mdp.m(3):prod(mdp.m(2:3))
                    P0(j,j) = 1; % tumor death 
                end

                % combine OAR/tumor matrix with surgery history
                if i == 1 % choose surgery
                    P1 = zeros(mdp.m(2)*mdp.m(3)); P1(:,end) = 1;
                    mdp.P(:,:,i) = kron([0,1],[P0;P1]);
                else % don't choose surgery
                    mdp.P(:,:,i) = kron(eye(2),P0);            
                end
            end
        end
        
        function probMat = calcProbMat(~,m,probVec,var)
        % Calculate the probability matrix for the given state variable.

            onesVec = cell(1,2);
            onesVec{1} = ones(1,m-1);
            onesVec{2} = ones(1,m);

            % diagonal elements
            probMat = zeros(m);
            for i = 1:3
                probMat = probMat + probVec(i)*diag(onesVec{2-abs(i-2)},i-2);
            end

            % boundary conditions
            if strcmp(var,'t')
                probMat(1,:) = [1 zeros(1,m-1)]; % tumor remission
                probMat(end,:) = zeros(1,m);     % tumor death
            else
                probMat(1,1) =  probVec(1) + probVec(2); % healthy OAR
                probMat(end,:) = zeros(1,m);             % OAR death
            end
        end
        
        function calcPolicy(mdp)
        % Calculate the optimal multi-modality policy.
        
            % compute immediate and terminal rewards
            S = mdp.calcStates();
            rt = mdp.calcReward(1,S);
            rN = mdp.calcReward(2,S);
            
            % compute transition probability matrix
            mdp.calcProb();

            % initialize optimal policy and patient utility
            mdp.A = zeros(mdp.numStates,mdp.N);
            V = rN;

            % compute optimal policy for each epoch with backward induction
            Vtemp = zeros(mdp.numStates,mdp.numActions);
            for t = mdp.N:-1:1
                % calculate maximum utility
                for a = 1:mdp.numActions
                    Vtemp(:,a) = mdp.P(:,:,a)*(rt+V);
                end
                V = max(Vtemp,[],2);
                
                % assign optimal modalities
                Atemp = zeros(mdp.numStates,1);
                for i = 1:mdp.numStates      
                    for j = 1:mdp.numActions
                        if abs(Vtemp(i,j)-V(i)) < mdp.tol
                            Atemp(i) = 10*Atemp(i)+j;
                        end
                    end
                end
                mdp.A(:,t) = Atemp;
            end
        end
        
        function printPolicy(mdp)
        % Print the optimal policy.

            for t = 1:mdp.N
                % surgery history = 0
                fprintf('Policy computed for h = 0 and t=%d\n',t)
                disp(flipud(reshape(mdp.A(1:mdp.m(2)*mdp.m(3),t),mdp.m(3),mdp.m(2))'))

                % surgery history = 1
                fprintf('Policy computed for h = 1 and t=%d\n',t)
                disp(flipud(reshape(mdp.A(mdp.m(2)*mdp.m(3)+1:end,t),mdp.m(3),mdp.m(2))'))
            end
        end
        
        function plotPolicy(mdp,leg)
        % Plot the optimal multi-modality policy for four actions.

            % assign actions to colormap
            map = mdp.defColormap();
            [B,labels,X] = mdp.action2color();

            % plot the policy for each epoch
            figure()
            for t = 1:mdp.N
                % surgery history = 0
                subplot(2,mdp.N+leg,t)
                imagesc(reshape(B(1:mdp.m(2)*mdp.m(3),t),mdp.m(3),mdp.m(2))')
                colormap(map)
                if mdp.numActions == 3
                    caxis([1 8])
                else
                    caxis([1 16])
                end
                title(sprintf('t=%d\n',t),'FontSize',14)
                set(gca,'YDir','normal'), axis square
                set(gca,'xtick',[]);
                if t == 1
                    ylabel({'s^h = 0','Side Effect'},'FontSize',14)
                else
                    set(gca,'ytick',[]);
                end

                % surgery history = 1
                subplot(2,mdp.N+leg,mdp.N+t+leg)
                imagesc(reshape(B(mdp.m(2)*mdp.m(3)+1:end,t),mdp.m(3),mdp.m(2))')
                colormap(map)
                if mdp.numActions == 3
                    caxis([1 8])
                else
                    caxis([1 16])
                end
                set(gca,'YDir','normal'), axis square
                xlabel('Tumor Progression','FontSize',14)
                if t == 1
                    ylabel({'s^h = 1','Side Effect'},'FontSize',14)
                else
                    set(gca,'ytick',[]);
                end
            end

            % display the legend
            if leg
                mdp.displayLegend(map,labels,X);
            end
            
            set(gcf,'Position',[10 500 800 350],...
                'PaperPositionMode','auto',...
                'Renderer','opengl');
        end
        
        function map = defColormap(mdp)
        % Define the policy colormap for three actions.
        
            map = [% new lines colormap
                   0.6806    0.0836    0.1972 % raspberry   (M1)
                   0.9957    0.7438    0.1340 % yellow      (M2)
                        0    0.4791    0.7942 % steel blue  (M3)
                   0.9110    0.3483    0.1050 % orange      (M1/M2)
                   0.3226    0.7985    1.0000 % sky blue    (M1/M3)
                   0.4995    0.7224    0.2015 % apple green (M2/M3)
                        0         0         0 % black       (M1/M2/M3)
                      0.8       0.8       0.8]; % white
        end
        
        function [B,labels,X] = action2color(mdp)
        % Assign actions to colormap colors for three actions.

            temp1 = [1 2 3 12 13 23 123];
            temp2 = {'M_1','M_2','M_3','M_1 or M_2',...
                'M_1 or M_3','M_2 or M_3','M_1, M_2, or M_3'};
            B = zeros(size(mdp.A)); labels = cell(1,length(temp1)); X = [];
            count = 1;
            for i = 1:length(temp1)
                if sum(sum(mdp.A == temp1(i))) > 0
                    B(mdp.A == temp1(i)) = i;
                    labels(count) = temp2(i);
                    X = [X; i length(temp1)+1];
                    count = count + 1;
                end
            end
            labels = labels(1:count-1);
        end
        
        function displayLegend(mdp,map,labels,X)
        % Display the modality legend.
        
            % plot the colormap
            subplot(2,mdp.N+1,[mdp.N+1 2*(mdp.N+1)])
            imagesc(X), axis off
            colormap(map)
            caxis([1 8])
  
            % annotate the legend
            for i = 1:length(labels)
                text(1.55,i,labels(i),'FontSize',14)
            end
            title('Modalities','FontSize',14)
        end
    end
end