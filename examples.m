% Examples from paper

clear all; close all; clc;

% 3.1: Base case (default parameters)

% Figure 1
mdp = mdpMain();
mdp.calcPolicy();
mdp.plotPolicy();

%% 3.2.1: Shape of terminal reward functions

% Figure 2
d = 3/2;
mdp.r{2} = @(S)100*((1/2)*(mdp.m(2)^d-S(2)^d)/mdp.m(2)^d+(1/2)*(mdp.m(3)^d-S(3)^d)/mdp.m(3)^d);
mdp.calcPolicy();
mdp.plotPolicy();

% Figure 3
d = 3;
mdp.r{2} = @(S)100*((1/2)*(mdp.m(2)^d-S(2)^d)/mdp.m(2)^d+(1/2)*(mdp.m(3)^d-S(3)^d)/mdp.m(3)^d);
mdp.calcPolicy();
mdp.plotPolicy();

%% 3.2.2: Relative importance of side effect and tumor progression

% Figure 4
c = 1/3;
mdp.r{2} = @(S)100*(c*(mdp.m(2)^2-S(2)^2)/mdp.m(2)^2+(1-c)*(mdp.m(3)^2-S(3)^2)/mdp.m(3)^2);
mdp.calcPolicy();
mdp.plotPolicy();

% Figure 5
c = 2/3;
mdp.r{2} = @(S)100*(c*(mdp.m(2)^2-S(2)^2)/mdp.m(2)^2+(1-c)*(mdp.m(3)^2-S(3)^2)/mdp.m(3)^2);
mdp.calcPolicy();
mdp.plotPolicy();


%% 3.3: Effect of intermediate rewards

mdp.r{2} = @(S)100*((1/2)*(mdp.m(2)^2-S(2)^2)/mdp.m(2)^2+(1/2)*(mdp.m(3)^2-S(3)^2)/mdp.m(3)^2);

% Figure 6
mdp.r{1} = @(S)1/4*100*(mdp.m(2)^2-S(2)^2)/mdp.m(2)^2;
mdp.calcPolicy();
mdp.plotPolicy();

% Figure 7
mdp.r{1} = @(S)1/4*100*(mdp.m(3)^2-S(3)^2)/mdp.m(3)^2;
mdp.calcPolicy();
mdp.plotPolicy();

%% 3.4: Effect of transition probabilities

mdp.r{1} = @(S)0;

% Figure 8
mdp.Pn{2,1} = [80 20 0]; % M1 tumor
mdp.calcPolicy();
mdp.plotPolicy();

% Figure 9
mdp.Pn{2,1} = [70 30 0]; % M1 tumor
mdp.Pn{1,2} = [0 70 30]; % M2 side-effect
mdp.calcPolicy();
mdp.plotPolicy();

% Figure 10
mdp.Pn{1,2} = [0 60 40]; % M2 side-effect
mdp.Pn{2,3} = [0 70 30]; % M3 tumor
mdp.calcPolicy();
mdp.plotPolicy();
