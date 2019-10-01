classdef mdpMain < handle
% mdpMain Compute the optimal multi-modality cancer treatment policy using
% backward induction for stationary transition probabilities.

    properties
       n          % number of state variables
       m          % number of states per state variable
       numStates  % total number of states
       numActions % number of actions
       T          % number of epochs
       r          % reward functions
       Pn         % side-effect and tumor transition probabilities
       P          % transition probability matrix
       A          % optimal policy
       tol        % tie threshold
    end
    
    methods
        % Initialize mdp variables according to pars and/or default values.
        function mdp = mdpMain(pars)
        
            flag = exist('pars','var');
            mdp.n = 3;
            if flag && isfield(pars,'m')
                mdp.m = pars.m;
            else
                mdp.m = [2,11,11];
            end
            mdp.numStates = prod(mdp.m);
            mdp.numActions = 3;
            if flag && isfield(pars,'T')
                mdp.T = pars.T;
            else
                mdp.T = 3;
            end
            if flag && isfield(pars,'r')
                mdp.r = pars.r;
            else
                mdp.r{1} = @(S)0;
                mdp.r{2} = @(S)100*((1/2)*(mdp.m(2)^2-S(2)^2)/mdp.m(2)^2+...
                    (1/2)*(mdp.m(3)^2-S(3)^2)/mdp.m(3)^2);
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
        
        % Check transition probability assumptions.
        function flag = checkAssumptions(mdp)
        
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
        
        % Create the matrix of all possible states.
        function S = calcStates(mdp)  
        
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
        
        % Calculate the immediate or terminal reward vector.
        function R = calcReward(mdp,t,S)
        
            rt = mdp.r{t};
            R = zeros(mdp.numStates,1);
            for i = 1:mdp.numStates
                R(i) = rt(S(i,:));
            end
        end
        
        % Calcuate the transition probability matrix.
        function calcProb(mdp)

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

                % combine OAR/tumor matrix with M1 history
                if i == 1 % choose M1
                    P1 = zeros(mdp.m(2)*mdp.m(3)); P1(:,end) = 1;
                    mdp.P(:,:,i) = kron([0,1],[P0;P1]);
                else % don't choose M1
                    mdp.P(:,:,i) = kron(eye(2),P0);            
                end
            end
        end
        
        % Calculate the probability matrix for the given state variable.
        function probMat = calcProbMat(~,m,probVec,var)

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
        
        % Calculate the optimal multi-modality policy.
        function calcPolicy(mdp)
        
            % compute immediate and terminal rewards
            S = mdp.calcStates();
            rt = mdp.calcReward(1,S);
            rN = mdp.calcReward(2,S);
            
            % compute transition probability matrix
            mdp.calcProb();

            % initialize optimal policy and patient utility
            mdp.A = zeros(mdp.numStates,mdp.T);
            V = rN;

            % compute optimal policy for each epoch with backward induction
            Vtemp = zeros(mdp.numStates,mdp.numActions);
            for t = mdp.T:-1:1
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
        
        % Print the optimal policy.
        function printPolicy(mdp)

            for t = 1:mdp.T
                % surgery history = 0
                fprintf('Policy computed for h = 0 and t=%d\n',t)
                disp(flipud(reshape(mdp.A(1:mdp.m(2)*mdp.m(3),t),mdp.m(3),mdp.m(2))'))

                % surgery history = 1
                fprintf('Policy computed for h = 1 and t=%d\n',t)
                disp(flipud(reshape(mdp.A(mdp.m(2)*mdp.m(3)+1:end,t),mdp.m(3),mdp.m(2))'))
            end
        end
        
        % Plot the optimal multi-modality policy for three actions.
        function p = plotPolicy(mdp)

            % assign actions to colormap
            map = mdp.defColormap();
            [B,labels,X] = mdp.action2color();

            % plot the policy for each epoch
            figure();
            res = 100;
            for t = 1:mdp.T
                % M1 history = 0
                h1 = subplot(2,mdp.T+1,t);
                p1 = get(h1,'pos');
                p1(1) = p1(1) - 0.03*(t-1);
                set(h1,'pos',p1);
                H0 = reshape(B(1:mdp.m(2)*mdp.m(3),t),mdp.m(3),mdp.m(2))';
                H0 = imresize(H0,res,'nearest');
                imagesc(H0);
                colormap(map)
                caxis([1 8])
                options = {'Interpreter','LaTeX','FontSize',13};
                temp = title(sprintf('t = %d',t));
                set(temp,options{:});
                set(gca,'YDir','normal'), axis square
                set(gca,'xtick',[]);
                if t == 1
                    set(gca,'ytick',res/2:2*res:10*res+res/2);
                    set(gca,'yticklabel',0:2:10);
                else
                    set(gca,'ytick',[]);
                    if t == 3
                        set(gca,'YAxisLocation','Right')
                        ylabel('h = 0',options{:})
                    end   
                end

                % M1 history = 1
                h2 = subplot(2,mdp.T+1,mdp.T+t+1);
                p2 = get(h2,'pos');
                p2(1) = p2(1) - 0.03*(t-1);
                p2(2) = p1(2) - p1(4) + 0.1;
                set(h2,'pos',p2);
                H1 = reshape(B(mdp.m(2)*mdp.m(3)+1:end,t),mdp.m(3),mdp.m(2))';
                H1 = imresize(H1,res,'nearest');
                imagesc(H1);
                colormap(map)
                caxis([1 8])
                set(gca,'YDir','normal'), axis square
                set(gca,'xtick',res/2:2*res:10*res+res/2);
                set(gca,'xticklabel',0:2:10);
                if t == 1
                    tx = text(8*res,-3*res,'Tumor Progression State ($$\tau$$)',options{:});
                    ty = text(-3.25*res,4.5*res,{'Side Effect State ($$\phi$$)'},options{:},'Rotation',90);
                    p = p2;
                    set(gca,'ytick',res/2:2*res:10*res+res/2);
                    set(gca,'yticklabel',0:2:10);
                else
                    set(gca,'ytick',[]);
                    if t == 3
                        set(gca,'YAxisLocation','Right')
                        ylabel('h = 1',options{:})
                    end    
                end
            end

            % plot the legend
            h3 = subplot(2,mdp.T+1,[mdp.T+1 2*(mdp.T+1)]);
            p3 = get(h3,'pos');
            p3(1) = p1(1) + p1(3) + 0.06;
            p3(2) = p2(2) + 0.065;
            p3(4) = p2(4) + 0.11;
            set(h3,'pos',p3);
            X = imresize(X,res,'nearest');
            imagesc(X), axis off
            colormap(map)
            caxis([1 8])

            % annotate the legend
            for i = 1:length(labels)
                text(1.1*res,(i-1/2)*res,labels(i),options{:})
            end
            text(p3(2)+0.15,p3(1)-0.25*res,'Modalities',options{:})
        end
        
        % Define the policy colormap for three actions.
        function map = defColormap(~)
        
            map = [0.6806    0.0836    0.1972   % raspberry   (M1)
                   0.9957    0.7438    0.1340   % yellow      (M2)
                        0    0.4791    0.7942   % steel blue  (M3)
                   0.9110    0.3483    0.1050   % orange      (M1/M2)
                   0.3226    0.7985    1.0000   % sky blue    (M1/M3)
                   0.4995    0.7224    0.2015   % apple green (M2/M3)
                        0         0         0   % black       (M1/M2/M3)
                   0.9400    0.9400    0.9400]; % gray
        end
        
        % Assign actions to colormap colors for three actions.
        function [B,labels,X] = action2color(mdp)

            temp1 = [1 2 3 12 13 23 123];
            temp2 = {'$$M_1$$','$$M_2$$','$$M_3$$','$$M_1$$ or $$M_2$$',...
                '$$M_1$$ or $$M_3$$','$$M_2$$ or $$M_3$$',...
                '$$M_1$$, $$M_2$$, or $$M_3$$'};
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
        
        % Display the modality legend.
        function displayLegend(mdp,map,labels,X)
        
            % plot the colormap
            subplot(2,mdp.T+1,[mdp.T+1 2*(mdp.T+1)])
            imagesc(X), axis off
            colormap(map)
            caxis([1 8])
  
            % annotate the legend
            for i = 1:length(labels)
                text(1.55,i,labels(i),'Interpreter','LaTeX','FontSize',14)
            end
            title('Modalities','Interpreter','LaTeX','FontSize',14)
        end
    end
end