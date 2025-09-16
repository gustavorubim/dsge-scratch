%--------------------------------------------------------------
%
% simple_simulate.m
%
% Simulate the dynamic response of the model to specific shocks
%
% by: Keith K�ster August 2003
%
%--------------------------------------------------------------

clear
addpath c:\wieland_project\simple_comm_bayes
dos('del c:\wieland_project\diagnostics\compute_aim_matrices.m')

modfile = ' c:\Wieland_project\simple_comm_bayes\simple_model.txt'
parnam  = ' simple_model_par'

rule = 1;       % 1: Taylor, 2: with smoothing, 3: Gerdesmeier, Roffia
% ----------------------
% set policy rule
% ----------------------
if rule ==1
    rho     = 0;
    alpha   = 1.5;
    beta    = 0.5;
elseif rule==2
    rho     = .79;
    alpha   = (1-.79)*2.2;
    beta    = (1-.79)*1.33;
elseif rule==3
    rho     =  .87^3;
    alpha   =  (1-.87^3)*1.93;
    beta    =  (1-.87^3)*.28;
else
    error('specify a rule between 1 and 3')
end


% ----------------------
% set parameters for the plot
% ----------------------

horizon = 30;

% ----------------------
% solve model for observable structure
% ----------------------




parseflag = 1;
solveflag = 1;

if solveflag == 1
   vwsolve
end

% ----------------------
% set up system matrices for dynamic simulations
% ----------------------

neq = size(cofb,1);

amat = cofb(1:neq,1:neq);  

bmat = inv(scof(1:neq,1*neq+1:2*neq));
bmat = bmat(1:neq,1:neq);

% number of periods
np = horizon+3;

% ----------------------
% initial conditions for endogenous variables
% ----------------------

x = zeros(np,neq);

 
% ----------------------
% initial conditions for exogenous shocks
% ----------------------

estn    = zeros(np,1);

%---------------------
% option to simulate dynamic
% response to one specific shock
%---------------------

not=0;
if not==0;
label = ['Simulate impulse response to:\n', ...
         '\n', ...
         '  1:  monetary shock\n', ...
         '\n', ...
         'Scenario (1-1):  '];
 
types = 0;               
while (~ismember(types,[1:11]) & isempty(types)) | ~ismember(types,[1:11])
   
   disp(' ')
   types = str2num(input(label,'s'));
   disp(' ')
end                
end


% -----------------------
% simulations start in period nstart, >=2
% a shock occurs in period nshock, >= nstart
% -----------------------

nstart = 2;
nshock = 2;
              
%---------------------
% define shocks
%---------------------

if types == 1
   estn(nshock) = .01;
end   

e = zeros(np,neq);
if rule ==1
e(:,loc(endog_,'stn'))    = estn*1.7950;
elseif rule ==2
e(:,loc(endog_,'stn'))    = estn*1.3112;
elseif rule ==3
e(:,loc(endog_,'stn'))    = estn*1.1348;
end
% ----------------------
% dynamic simulation from initial conditions
% ----------------------

for i = nstart:np
   x1 = x(i-1,1:neq)';
   ee = e(i,:)';
   xc = amat*x1+bmat*ee;
   x(i,1:neq) = xc';
end


% ----------------------
% preparation for plotting and printout
% ----------------------

nprint = np;

n = nstart:nprint;

irpinf = x(nstart:nprint,loc(endog_,'pipcda'))*100;
irq    = x(nstart:nprint,loc(endog_,'yer'))*100;
irr    = x(nstart:nprint,loc(endog_,'stn'))*100; % to scale from quarterly to annual percent, 



clf;
subplot(1,3,1);hold on; plot1=plot(n,irpinf,'-'); 
title('Annual Inflation','Fontsize', 16, 'VerticalAlignment', 'bottom')
                line('XData',[nstart, nprint],'YData',[0, 0], 'LineStyle','-.'), 
                axis tight;
                xlabel('Quarters', 'Fontsize', 9);    
                set(gca,'XTick', [nstart:5:200], ...
                    'XTickLabel', num2str([0:5:200]'), 'FontSize', 10)
                % ----------------------------------
                % Depending on the rule chosen, set 
                % the bounds of the graphs
                % ----------------------------------

                
                hold off;
                
subplot(1,3,2);hold on; plot2=plot(n,irq,'.-');
title('Output Gap','Fontsize', 16, 'VerticalAlignment', 'bottom')

                line('XData',[nstart, nprint],'YData',[0, 0], 'LineStyle','-.'),  
                       axis tight; %axis([1 21 -.3 0.4]);
                xlabel('Quarters', 'Fontsize', 9); 
                set(gca,'XTick', [nstart:5:200], ...
                    'XTickLabel', num2str([0:5:200]'), 'FontSize', 10)
                hold off;

                
subplot(1,3,3);hold on; plot3=plot(n,irr,'.-');  
title('Nominal Rate','Fontsize', 16, 'VerticalAlignment', 'bottom')
                line('XData',[nstart, nprint],'YData',[0, 0], 'LineStyle','-.'),  
                        axis tight;%axis([1 21 -.6 0.025]);
                xlabel('Quarters', 'Fontsize', 9);
                set(gca,'XTick', [nstart:5:200], ...
                    'XTickLabel', num2str([0:5:200]'), 'FontSize', 10)
                hold off;
                

if      rule==1 & types==1
    irpinf_simple = irpinf;
    irq_simple    = irq;
    irr_simple    = irr;
    save c:\wieland_project\impulse_data\simple_Taylor_IR irpinf_simple irq_simple irr_simple
elseif rule==2 & types==1
    irpinf_simple = irpinf;
    irq_simple    = irq;
    irr_simple    = irr;
    save c:\wieland_project\impulse_data\simple_smooth_IR irpinf_simple irq_simple irr_simple
elseif rule==3 & types==1
    irpinf_simple = irpinf;
    irq_simple    = irq;
    irr_simple    = irr;
    save c:\wieland_project\impulse_data\simple_Gerdes_IR irpinf_simple irq_simple irr_simple
end

rmpath c:\wieland_project\simple_comm_bayes