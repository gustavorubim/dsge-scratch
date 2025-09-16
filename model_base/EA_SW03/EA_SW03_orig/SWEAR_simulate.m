%--------------------------------------------------------------
%
% SWEAR_simulate.m
%
% Simulate the dynamic response of the model to specific shocks
%
% by: Keith K�ster August 2003
%
%--------------------------------------------------------------

% clear

% -----------------------------------------
% Define the parameter name and modelfile
% -----------------------------------------

thispath = cd;


solve_path = horzcat(thispath,'\newaim'); % path of sp_solve directory on your machine
addpath(solve_path);
dirnam= horzcat(' ',thispath,'\'); % directory of this example 

addpath(thispath)
horzcat('dos('' del ', thispath, '\compute_aim_matrices.m '' ',')')

modnam = 'Swear_old.txt';%' 
parnam  = ' SWEAR_par'%;_ACS'%';%
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

horizon = 40;

% ----------------------
% solve model for observable structure
% ----------------------




parseflag = 1;
solveflag = 1;

if solveflag == 1
   spsolve
end

% -------

% ----------------------
% set up system matrices for dynamic simulations
% ----------------------

neq = size(cofb,1);

amat = cofb(1:neq,1:neq);  

bmat = inv(scof(1:neq,1*neq+1:2*neq));
bmat = bmat(1:neq,1:neq);


% ---------------------
% Construct Omega
% ---------------------
Sigma_e = zeros(neq,neq);
Sigma_e(loc(endog_,'ea'),   loc(endog_,'ea'   )) = 1;
Sigma_e(loc(endog_,'eb'),   loc(endog_,'eb'   )) = 1;
Sigma_e(loc(endog_,'eg'),   loc(endog_,'eg'   )) = 1;
Sigma_e(loc(endog_,'els'),  loc(endog_,'els'  )) = 1;
Sigma_e(loc(endog_,'eqs'),  loc(endog_,'eqs'  )) = 1;
Sigma_e(loc(endog_,'eps'),  loc(endog_,'eps'  )) = 1;
Sigma_e(loc(endog_,'em'),   loc(endog_,'em'   )) = 1;
Sigma_e(loc(endog_,'eas'),  loc(endog_,'eas'  )) = 1;
Sigma_e(loc(endog_,'econs'),loc(endog_,'econs')) = 1;
Sigma_e(loc(endog_,'einv'), loc(endog_,'einv' )) = 1;
Sigma_e(loc(endog_,'ey'),   loc(endog_,'ey'   )) = 1;
Sigma_e(loc(endog_,'elab'), loc(endog_,'elab' )) = 1;
Sigma_e(loc(endog_,'epinf'),loc(endog_,'epinf')) = 1;
Sigma_e(loc(endog_,'ew'),   loc(endog_,'ew'   )) = 1;

% ---------------------
% Autocorrelation functions
% ---------------------
index = [loc(endog_,'pinf4'),loc(endog_,'ygap'),loc(endog_,'dr')];
[COR, Std]   = autocorrelation_2(amat,bmat,80,Sigma_e,index) 
save Swear_autocorr endog_ index COR Std rule



% number of periods
np = horizon+3;

% ----------------------
% initial conditions for endogenous variables
% ----------------------

x = zeros(np,neq);

 
% ----------------------
% initial conditions for exogenous shocks
% ----------------------

ea0    = zeros(np,1);
eb0    = zeros(np,1);
eg0    = zeros(np,1);
einv0  = zeros(np,1);
els0   = zeros(np,1);
em0    = zeros(np,1);
epinf0 = zeros(np,1);
eqs0   = zeros(np,1);
ew0    = zeros(np,1);
eas0   = zeros(np,1);

%---------------------
% option to simulate dynamic
% response to one specific shock
%---------------------

not=0;
if not==0;
label = ['Simulate impulse response to:\n', ...
         '\n', ...
         '  1:  technology shock\n', ...
         '  2:  preference shock\n', ...
         '  3:  fiscal shock\n', ...
         '  4:  investment shock\n', ...
         '  5:  labor supply shock\n', ...
         '  6:  monetary policy shock\n', ...
         '  7:  price shock\n', ...
         '  8:  shock to Tobin''s Q\n', ...
         '  9:  wage shock\n', ...
         ' 10:  shock to inflation target\n', ...
         '\n', ...
         'Scenario (1-10):  '];
 
types = 0;               
while (~ismember(types,[1:10]) & isempty(types)) | ~ismember(types,[1:10])
   
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
   ea0(nshock) = 1.0;
elseif types == 2
   eb0(nshock) = 1.00;
elseif types == 3
   eg0(nshock) = 1.00;
elseif types == 4
   einv0(nshock) = 1.00;
elseif types == 5
   els0(nshock) = 1.00;
elseif types == 6
   if rule==1
    em0(nshock) = .25*1.0264 *100;%* 12.7035;
   elseif rule==2
    em0(nshock) = .25*1.0552;
   elseif rule==3
    em0(nshock) = 0.0808*.25*4*3.1479;% set =1 if compare estimated rule to cambridge paper
   end
   %em0(nshock) = 1;%*1.4308;
elseif types == 7
   epinf0(nshock) = 1.00;
elseif types == 8
   eqs0(nshock) = 1.00;
elseif types == 9
   ew0(nshock) = 1.00;
elseif types == 10
   eas0(nshock) = 1.00;
end   

e = zeros(np,neq);
e(:,loc(endog_,'ea'))    = ea0;
e(:,loc(endog_,'eb'))    = eb0;
e(:,loc(endog_,'eg'))    = eg0;
e(:,loc(endog_,'einv'))  = einv0;
e(:,loc(endog_,'els'))   = els0;
e(:,loc(endog_,'r'))    = em0;   % type e(:,loc(endog_,'em'))    = em0; if compare to Cambridge paper
e(:,loc(endog_,'epinf')) = epinf0;
e(:,loc(endog_,'eqs'))   = eqs0;
e(:,loc(endog_,'ew'))    = ew0;
e(:,loc(endog_,'eas'))   = eas0;

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

irpinf = x(nstart-1:nprint-1,loc(endog_,'pinf4'))*1;%;
irq    = x(nstart-1:nprint-1,loc(endog_,'ygap'));
irr    = x(nstart-1:nprint-1,loc(endog_,'r'))*4;%; % first *4 to scale from quarterly to annual percent, second 100 to scale to bps




% clf;
figure;
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
                if rule==1
                    set(gca,'YTick', [-0.04:.01:.04], ...
                    'YTickLabel', num2str([-0.04:.01:.04]'), 'FontSize', 10)
                    axis([n(1) n(end) -.02 0.011])
                end
                
                hold off;
                
subplot(1,3,2);hold on; plot2=plot(n,irq,'.-');
title('Output Gap','Fontsize', 16, 'VerticalAlignment', 'bottom')

                line('XData',[nstart, nprint],'YData',[0, 0], 'LineStyle','-.'),  
                       axis tight; %axis([1 21 -.3 0.4]);
                xlabel('Quarters', 'Fontsize', 9); 
                set(gca,'XTick', [nstart:5:200], ...
                    'XTickLabel', num2str([0:5:200]'), 'FontSize', 10)
                hold off;
                if rule==1
                     set(gca,'YTick', [-0.04:.01:.04], ...
                    'YTickLabel', num2str([-0.04:.01:.04]'), 'FontSize', 10)
                    axis([n(1) n(end) -.02 0.011])
                end
                
subplot(1,3,3);hold on; plot3=plot(n,irr,'.-');  
title('Nominal Rate','Fontsize', 16, 'VerticalAlignment', 'bottom')
                line('XData',[nstart, nprint],'YData',[0, 0], 'LineStyle','-.'),  
                        axis tight;%axis([1 21 -.6 0.025]);
                xlabel('Quarters', 'Fontsize', 9);
                set(gca,'XTick', [nstart:5:200], ...
                    'XTickLabel', num2str([0:5:200]'), 'FontSize', 10)
                hold off;
                if rule==1
                     set(gca,'YTick', [-10:10:100], ...
                    'YTickLabel', num2str([-10:10:100]'), 'FontSize', 10)
                    axis([n(1) n(end) -10 100])
                end
         

if      rule==1 & types==6
    irpinf_SWEAR = irpinf;
    irq_SWEAR    = irq;
    irr_SWEAR    = irr;
    %save c:\wieland_project\impulse_data\Swear_Taylor_IR irpinf_SWEAR irq_SWEAR irr_SWEAR 
elseif rule==2 & types==6
    irpinf_SWEAR = irpinf;
    irq_SWEAR    = irq;
    irr_SWEAR    = irr;
    %save c:\wieland_project\impulse_data\Swear_smooth_IR irpinf_SWEAR irq_SWEAR irr_SWEAR 
elseif rule==3 & types==6
    irpinf_SWEAR = irpinf;
    irq_SWEAR    = irq;
    irr_SWEAR    = irr;
    %save c:\wieland_project\impulse_data\Swear_Gerdes_IR irpinf_SWEAR irq_SWEAR irr_SWEAR 
end

%rmpath c:\wieland_project\simple_comm_bayes