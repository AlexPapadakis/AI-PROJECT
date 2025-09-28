%% Main script for regression project

clc; clear; close all;

% Load saved workspace after training
disp('Loading saved regression workspace...');
load('project3A.mat');

% Now run plotting
disp('Generating plots...');
run('regressionA_plotting.m');

disp('All done.');
