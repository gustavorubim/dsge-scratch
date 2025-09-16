% Calling program for 2 parameter rule optimization
% by John C. Williams (10/27/97)
%

%
% Declare model file, parameter file, and directory name for AIM
%

modnam='ms97jcw';
parnam='mspar2';
dirnam='johnw/msmodel';
parseflag=0;

wt1=0;
wt3=0;

% load varcov matrix for shocks

load shocks97;

% Zero out funds rate, foreign shocks

shocks97(:,1)=0*shocks97(:,1);
%shocks97(:,6)=0*shocks97(:,6);

vcovmat=cov(shocks97(:,1:7));
clear shocks97;

% shocks:
%  1:  rffsh  
%  2:  ecsh   
%  3:  efish  
%  4:  eiish  
%  5:  egsh   
%  6:  exsh   
%  7:  cwsh   


tayr1	=	0.79504065;
tayp0	=       (.624517214 -(1-0.79504065));
tayx0	=       1.171022658;
tayx1	=       -0.966970286;
tayx2	=	0;


tayr1	=	1;
tayp0	=	0.5;
tayx0	=	0.5;
tayx1	=	0;
tayx2	=	0;

tayr1	=	0.79504065;
%tayr1	=	0.75904065;
tayp0	=       (.624517214 -(1-0.79504065));
tayx0	=       1.171022658;
tayx1	=       -0.966970286;
tayx2	=	0;

tayr1	=	.755226;
tayp0	=       (.602691 -(1-tayr1))/4;
tayx0	=       1.17616;
tayx1	=       -.972390;
tayx2	=	0;

tayr1	=	0;
tayp0	=	0;
tayx0	=	0;
tayx1	=	0;
tayx2	=	0;

mysolvenewmod;
msuncond;

%msrest;

%msrauto;
  [tayr1,taypl4,tayx0,100*sdygap,100*sdpdot,100*sddrff]

interres=zeros(1,7);
if aimcode==1,
  interres(1)=100*sdygap;
  interres(2)=100*sdpdot;
  interres(3)=100*sddrff;
end
  interres(4)=tayr1;
  interres(5)=taypm4;
  interres(6)=tayx0;
  interres(7)=aimcode;

fid=fopen('/mq/home/m1jcw99/models/ms/flrules/msrjunk.dat','a');
fprintf(fid,'%12.8f %12.8f %12.8f %8.4f %8.4f %8.4f %6.0f\n',interres);                       

