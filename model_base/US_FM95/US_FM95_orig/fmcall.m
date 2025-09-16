clear; clc; close all; warning off; tic

thispath = cd;


solve_path = horzcat(thispath,'\newaim'); % path of sp_solve directory on your machine
addpath(solve_path);
dirnam= horzcat(' ',thispath,'\'); % directory of this example 

addpath(thispath)
horzcat('dos('' del ', thispath, '\compute_aim_matrices.m '' ',')')

% setup aim inputs

modnam='fmjmcbf_CHANGE_MAIK';
parnam='fmpar';
cofintintb1 =  0.76;
cofintinf0 = 0.6;
cofintout = 1.18; cofintoutb1 = -0.97;

ntestmat=3;

% load varcov matrix for shocks

load fmomega.dat;
vcovmat=fmomega;

% no funds rate shocks

vcovmat(:,2)=zeros(3,1);
vcovmat(2,:)=zeros(1,3);


tayr1	=	.755226;
tayp0	=       (.602691 -(1-tayr1))/4;
tayp1	=       tayp0;
tayp2   =       tayp0;
tayp3	=       tayp0;
tayp4	=	0;
tayx0	=       1.17616;
tayx1	=       -.972390;

tayr1 = 0;
taypm4=.0;
taypm3	=       0;
taypm2   =       0;
taypm1	=       0;
tayp0=.5/4;
tayp1=tayp0;
tayp2=tayp0;
tayp3=tayp0;
tayp4 = 0;
tayxl1=0;
tayx0 =       .5;
tayx1=0;


%mysolvenew;

% Solve model
parseflag = 1;
solveflag = 1;

spsolve

fmuncond;

[tayr1,taypm4,tayx0,100*sdygap,100*sdpdot,100*sddrff]

interres=zeros(12,1);

wt1=0;
wt3=0;

  interres(1)=100*sdygap;
  interres(2)=100*sdpdot;
  interres(3)=100*sdrff ;
  interres(4)=100*sddrff;
  interres(5)=tayr1;
  interres(6)=taypm4;
  interres(7)=tayx0;
  interres(8)=aimcode;
  interres(9)=wt1;
  interres(10)=wt3;
  interres(11)=0;
  interres(12)=0;


% write parameter values/outcomes to file 

%fid=fopen('/mq/home/m1jcw99/models/fm/flrules/fmjunk.dat','a');
%fprintf(fid,'%12.8f %12.8f %12.8f %12.8f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f \n',interres);    

%modelest;

%fmauto;


