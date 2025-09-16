var y c k i a;
varexo e;
parameters alpha beta delta rho;

alpha = 0.36;
beta = 0.99;
delta = 0.025;
rho = 0.9;

model(linear);
  y = c + i;
  c = (1 - alpha) * y;
  i = y - c;
  k = (1 - delta) * k(-1) + i;
  a = rho * a(-1) + e;
end;

initval;
  y = 1;
  c = 0.8;
  i = 0.2;
  k = 10;
  a = 0;
end;

shocks;
  var e;
  stderr 0.01;
end;

varobs y c i;
