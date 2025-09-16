%--------------------------------------------------------------
%
% AWM_simulate.m
%
% Simulate the dynamic response of the model to specific shocks
%
% by: Keith K�ster August 2003
%
%--------------------------------------------------------------

clear
% -----------------------------------------
% Define the parameter name and modelfile
% -----------------------------------------

thispath = cd;


solve_path = horzcat(thispath,'\newaim'); % path of sp_solve directory on your machine
addpath(solve_path);
dirnam= horzcat(' ',thispath,'\'); % directory of this example 

addpath(thispath)
horzcat('dos('' del ', thispath, '\compute_aim_matrices.m '' ',')')


modnam = 'AWM2.txt'
%modfile = ' c:\Wieland_project\simple_comm_bayes\AWM2.txt'
parnam  = ' AWM_par'

rule = 3;       % 1: Taylor, 2: with smoothing, 3: Gerdesmeier, Roffia
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

horizon = 50;

% ----------------------
% solve model for observable structure
% ----------------------




parseflag = 1;
solveflag = 1;

% change by VW 16.8.06
%if solveflag == 1
%   vwsolve
%end
if solveflag == 1
   spsolve
end
%%
% ----------------------
% set up system matrices for dynamic simulations
% ----------------------

neq = size(cofb,1);

amat = cofb(1:neq,1:neq);  

bmat = inv(scof(1:neq,1*neq+1:2*neq));
bmat = bmat(1:neq,1:neq);


% ---------------------
% Autocorrelation functions
% ---------------------
index = [loc(endog_,'pipcda'),loc(endog_,'yer'),loc(endog_,'stn')];
[COR, Std]   = autocorrelation_2(amat,bmat(:,end-13:end),80,Omega,index) 
%save AWM_autocorr endog_ index COR Std rule



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
   estn(nshock) = 1.0;
end   

e = zeros(np,neq);
if rule ==1
e(:,loc(endog_,'stn'))    = estn*1.0322;
elseif rule ==2
e(:,loc(endog_,'stn'))    = estn*1.0185;
elseif rule ==3
e(:,loc(endog_,'stn'))    = estn*1.0124;
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

irpinf = x(nstart-1:nprint-1,loc(endog_,'pipcda'))*100;
irq    = x(nstart-1:nprint-1,loc(endog_,'yer'))*100;
irr    = x(nstart-1:nprint-1,loc(endog_,'stn')); % to scale from quarterly to annual percent, 




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
    irpinf_AWM = irpinf;
    irq_AWM    = irq;
    irr_AWM    = irr;
    %save c:\wieland_project\impulse_data\AWM_Taylor_IR irpinf_AWM irq_AWM irr_AWM
elseif rule==2 & types==1
    irpinf_AWM = irpinf;
    irq_AWM    = irq;
    irr_AWM    = irr;
    %save c:\wieland_project\impulse_data\AWM_smooth_IR irpinf_AWM irq_AWM irr_AWM
elseif rule==3 & types==1
    irpinf_AWM = irpinf;
    irq_AWM    = irq;
    irr_AWM    = irr;
    %save c:\wieland_project\impulse_data\AWM_Gerdes_IR irpinf_AWM irq_AWM irr_AWM
end

rmpath c:\wieland_project\simple_comm_bayes