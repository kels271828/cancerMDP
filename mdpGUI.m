classdef mdpGUI < handle
% mdpGUI Compute the optimal multi-modality cancer treatment policy using
% backward induction for stationary transition probabilities.
%
% Note: In general we assume that the normal tissue side effect can only 
% stay the same or get worse when treating the cancer, while it can only 
% stay the same or get better when choosing not to treat the cancer. 
% Similarly, we assume that the tumor progression can only stay the same
% or get better when treating the cancer, while it can ony stay the same
% or get worse when choosing not to treat the cancer.

    properties
       mdp  % mdp object
       pars % gui parameters
       func % reward function parameters
    end
    
    methods
        function gui = mdpGUI(pars)
        % Initialize mdp parameters and create GUI.
        
            % initialize mdp variables
            if exist('pars','var')
                gui.mdp = mdpMain(pars);
            else
                gui.mdp = mdpMain();
            end
            
            % set GUI parameters
            gui.pars = gui.setPars();
            
            % set function parameters
            gui.func = gui.setFunc();
            
            % create and then hide the GUI as it is begin constructed
            f = figure('Visible','off','Position',gui.pars.f);
            
            % add labels
            background = axes('Position',[0 0 1 1],...
                'Color','none');
            set(background,...
                'XColor',get(gcf,'Color'),...
                'YColor',get(gcf,'Color'));
            text(125,200,'State Transition Probabilities','Units','Pixels');
            text(460,200,'Reward Functions','Units','Pixels');  
            text(380,150,'r_t =             f(\phi;        ) +         g(\tau;        )','Units','Pixels','FontSize',15);
            text(380,110,'r_{T+1} =        f(\phi;        ) +         g(\tau;        )','Units','Pixels','FontSize',15);
             
            % create transition probability input tables
            mLabels = cell(1,gui.mdp.numActions);
            mLabels2 = cell(1,2);
            pTables = cell(2,gui.mdp.numActions);
            for i = 1:gui.mdp.numActions
                pos = gui.pars.m; pos(1) = pos(1)+(i-1)*gui.pars.column;
                text(pos(1),pos(2),gui.pars.mStrings{i},'Units','Pixels')
                
                pos = gui.pars.m1;
                pos(1) = pos(1)+20;
                pos(1) = pos(1)+(i-1)*gui.pars.column;
                text(pos(1),pos(2)+10,gui.pars.mStrings2{1},'Units','Pixels');
                
                pos = gui.pars.m2;
                pos(1) = pos(1)+5;
                pos(1) = pos(1)+(i-1)*gui.pars.column;
                text(pos(1),pos(2)+10,gui.pars.mStrings2{2},'Units','Pixels');
            
                pTables{1,i} = uitable('CreateFcn',{@gui.pCreate,1,i});
                pTables{2,i} = uitable('CreateFcn',{@gui.pCreate,2,i});
            end
           

            % create inputs for epochs, discount factor, and reward functions
            scaleInputs1 = cell(1,2);
            scaleInputs2 = cell(1,2);
            expInputs1 = cell(1,2);
            expInputs2 = cell(1,2);
            for i = 1:2
                scaleInputs1{i} = uicontrol('CreateFcn',{@gui.scaleCreate1,i});
                expInputs1{i}   = uicontrol('CreateFcn',{@gui.expCreate1,i});
                scaleInputs2{i} = uicontrol('CreateFcn',{@gui.scaleCreate2,i});
                expInputs2{i}   = uicontrol('CreateFcn',{@gui.expCreate2,i});
            end
           
            % compute policy button
            button = uicontrol('Callback',@gui.bCallback,...
                'Position',gui.pars.b,...
                'String','Compute policy',...
                'Style','pushbutton');
            
            % assign the GUI a name to appear in the window title
            set(f,'Name','Multi-Modality MDP')
            
            % move the GUI to the center of the screen
            movegui(f,'center')
            
            % make the GUI visible
            set(f,'Visible','on'); 
        end
        
        function pars = setPars(gui)
        % Set GUI parameters.
        
            % width and height
            s2W = 94;  s2H = 39; % tumor transition input tables
            s1W = 94;  s1H = 39; % OAR transition input tables
            mW  = 20;  mH  = 20; % input table labels
            m1W = s2W; m1H = 15;
            m2W = s2W; m2H = 15;
            bW  = 100;  bH  = 50; % calc policy button

            % spacing
            pars.border = 25;
            pars.row = 30;
            pars.column = pars.border+s2W;

            % bottom and left
            s2B = pars.border; s2L = pars.border; 
            m2B = s2B+s2H;     m2L = pars.border;
            s1B = s2B+s2H+m2H+15;  s1L = pars.border; 
            m1B = s1B+s1H;     m1L  = pars.border;
            mB  = s1B+s1H+2*m1H;  mL  = pars.border+s2W/2-10;       
            bB  = s2B;    bL  = pars.border+gui.mdp.numActions*pars.column;
            
            % set alignment parameters
            pars.f  = [500 500 650 225];
            pars.m  = [mL mB mW mH];
            pars.m1 = [m1L m1B m1W m1H];
            pars.m2 = [m2L m2B m2W m2H];
            pars.s1 = [s1L s1B s1W s1H];
            pars.s2 = [s2L s2B s2W s2H];
            pars.b  = [bL bB bW bH];
            
            % set labels and functions
            pars.mStrings = {'M_1','M_2','M_3'};
            pars.mStrings2 = {'Side Effect','Tumor Progression'};
        end
        
        function func = setFunc(gui)
        % Set reward function parameters.
        
            % side effect function parameters
            func.c = {0,1/3}; % scale coefficient
            func.p = {2,2};   % exponent
            
            % tumor progression function parameters
            func.d = {0,2/3}; % scale coefficient
            func.q = {2,2};   % exponent
            
            % set individual functions
            func.f = @(c,p,S) 100*c*(gui.mdp.m(2)^p-S(2)^p)/gui.mdp.m(2)^p;
            func.g = @(d,q,S) 100*d*(gui.mdp.m(3)^q-S(3)^q)/gui.mdp.m(3)^q;
            
            % set combined reward functions
            gui.mdp.r{1} = @(S) func.f(func.c{1},func.p{1},S)+...
                func.g(func.d{1},func.q{1},S);
            gui.mdp.r{2} = @(S) func.f(func.c{2},func.p{2},S)+...
                func.g(func.d{2},func.q{2},S);
        end
        
        function scaleCreate1(gui,hObject,~,i)
            % create input
            if i == 1
                y = 140;
            else
                y = 100;
            end
            set(hObject,...
                'Callback',{@gui.cCallback,i},...
                'Position',[425 y 25 25],...
                'String',gui.func.c{i},...
                'Style','edit');
        end
        
        function expCreate1(gui,hObject,~,i)
            % create input
            if i == 1
                y = 140;
            else
                y = 100;
            end
            set(hObject,...
                'Callback',{@gui.ppCallback,i},...
                'Position',[479 y 25 25],...
                'String',gui.func.p{i},...
                'Style','edit');
        end
        
        function scaleCreate2(gui,hObject,~,i)
            % create input
            if i == 1
                y = 140;
            else
                y = 100;
            end
            set(hObject,...
                'Callback',{@gui.dCallback,i},...
                'Position',[530 y 25 25],...
                'String',gui.func.d{i},...
                'Style','edit');
        end
        
        function expCreate2(gui,hObject,~,i)
            % create input
            if i == 1
                y = 140;
            else
                y = 100;
            end
            set(hObject,...
                'Callback',{@gui.qCallback,i},...
                'Position',[586 y 25 25],...
                'String',gui.func.q{i},...
                'Style','edit');
        end
        
        function pCreate(gui,hObject,~,i,j)
        % Create transition probability input table.
        
            % set labels and position
            if i == 1
                pos = gui.pars.s1;
                colNames = {'-1','=','+1'};
            else
                pos = gui.pars.s2; 
                colNames = {'-1','=','+1'};
            end
            pos(1) = pos(1)+(j-1)*gui.pars.column;
            
            % create table
            set(hObject,...
                'CellEditCallback',{@gui.pCallback,i,j},...
                'ColumnEditable',true,...
                'ColumnName',colNames,...
                'ColumnWidth',{30 30 30},...
                'Data',gui.mdp.Pn{i,j},...
                'Position',pos,...
                'RowName',{});
        end
        
        function cCallback(gui,hObject,~,i)
            newC = str2double(get(hObject,'String'));
            gui.func.c{i} = newC;
            gui.mdp.r{i} = @(S) gui.func.f(gui.func.c{i},gui.func.p{i},S)+...
            	gui.func.g(gui.func.d{i},gui.func.q{i},S);
        end
        
        function dCallback(gui,hObject,~,i)
            newD = str2double(get(hObject,'String'));
            gui.func.d{i} = newD;
            gui.mdp.r{i} = @(S) gui.func.f(gui.func.c{i},gui.func.p{i},S)+...
            	gui.func.g(gui.func.d{i},gui.func.q{i},S);
        end
        
        function ppCallback(gui,hObject,~,i)
            newP = str2double(get(hObject,'String'));
            gui.func.p{i} = newP;
            gui.mdp.r{i} = @(S) gui.func.f(gui.func.c{i},gui.func.p{i},S)+...
            	gui.func.g(gui.func.d{i},gui.func.q{i},S);
        end
        
        function qCallback(gui,hObject,~,i)
            newQ = str2double(get(hObject,'String'));
            gui.func.q{i} = newQ;
            gui.mdp.r{i} = @(S) gui.func.f(gui.func.c{i},gui.func.p{i},S)+...
            	gui.func.g(gui.func.d{i},gui.func.q{i},S);
        end
        
        function pCallback(gui,hObject,eventdata,i,j)
        % Update transition probabilities.
            
            % store values
            newP = str2double(eventdata.EditData);
            oldP = gui.mdp.Pn{i,j};
            temp = oldP;
            
            % replace and check
            row = eventdata.Indices(1);
            col = eventdata.Indices(2);
            temp(row,col) = newP;
            gui.mdp.Pn{i,j} = temp;
            flag = gui.mdp.checkAssumptions();
            
            % if invalid values
            if flag || ~isfinite(newP)
                gui.mdp.Pn{i,j} = oldP;
                set(hObject,'Data',oldP);
            end
        end
        
        function bCallback(gui,~,~)
        % Compute optimal policy.
            
            gui.mdp.calcPolicy();
            gui.mdp.plotPolicy(1); % 1 = display legend
        end
    end
end
