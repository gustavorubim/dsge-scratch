%********************************************************
%Run and plot IRFs for Ireland (2004) for:
% a preference shock (epsa_)
% a technolgy shock (epsz_)
% a monetary policy shock (interest_)
%********************************************************

clear all;
clc;
close all;

%adjust path to folder where replication file is stored
cd([cd '/NK_IR04_rep']);

%run replication dynare file
dynare NK_IR04_rep;

%load results
load NK_IR04_rep_results.mat;

nul=zeros(17,1);
t0=0:1:16;
t=0:1:16;

y_epsa_=[0;y_epsa_];
y_epsz_=[0;y_epsz_];
y_interest_=[0;y_interest_];

m_epsa_=[0;m_epsa_];
m_epsz_=[0;m_epsz_];
m_interest_=[0;m_interest_];

pi_epsa_=[0;pi_epsa_];
pi_epsz_=[0;pi_epsz_];
pi_interest_=[0;pi_interest_];

r_epsa_=[0;r_epsa_];
r_epsz_=[0;r_epsz_];
r_interest_=[0;r_interest_];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot IRFs for the variables:
%   output (y),
%   real money (m),
%   inflation (pi),
%   interest rate (r) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('name','Impulse responses','numbertitle','off')

%Preference shock
subplot(3,4,1);
plot(t,y_epsa_,'b','LineWidth',1.5);           hold on          
plot(t0,nul,'r');
axis([0 16 0 0.6]);
title('Output to Preference Shock');

subplot(3,4,2);
plot(t,m_epsa_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 -0.2 0.05]);
title('Real Money to Preference Shock');

subplot(3,4,3);
plot(t,pi_epsa_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 0 0.3]);
title('Inflation to Preference Shock');

subplot(3,4,4);
plot(t,r_epsa_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 0 0.25]);
title('Interest Rate to Preference Shock');

%Technology shock
subplot(3,4,5);
plot(t,y_epsz_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 0 1]);
title('Output to Technology Shock');

subplot(3,4,6);
plot(t,m_epsz_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 0 0.06]);
title('Real Money to Technology Shock');

subplot(3,4,7);
plot(t,pi_epsz_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 -0.05 0.01]);
title('Inflation to Technology Shock');

subplot(3,4,8);
plot(t,r_epsz_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 -0.5 0.01]);
title('Interest Rate to Technology Shock');


%Monetary Policy Shock  
subplot(3,4,9);
plot(t,y_interest_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 -0.5 0.1]);
title('Interest Rate to Preference');

subplot(3,4,10);
plot(t,m_interest_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 -0.2 0.05]);
title('Interest Rate to Cost-Push');

subplot(3,4,11);
plot(t,pi_interest_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 -0.08 0.02]);
title('Interest Rate to Technology');

subplot(3,4,12);
plot(t,r_interest_,'b','LineWidth',1.5);              hold on
plot(t0,nul,'r');
axis([0 16 0 0.3]);
title('Interest Rate to Monetary Policy');

