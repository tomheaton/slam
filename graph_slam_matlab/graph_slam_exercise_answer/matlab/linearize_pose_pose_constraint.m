% Compute the error of a pose-pose constraint
% x1 3x1 vector (x,y,theta) of the first robot pose
% x2 3x1 vector (x,y,theta) of the second robot pose
% z 3x1 vector (x,y,theta) of the measurement
%
% You may use the functions v2t() and t2v() to compute
% a Homogeneous matrix out of a (x, y, theta) vector
% for computing the error.
%
% Output
% e 3x1 error of the constraint
% A 3x3 Jacobian wrt x1
% B 3x3 Jacobian wrt x2
function [e, A, B] = linearize_pose_pose_constraint(x1, x2, z)

  % compute the transformation from the vectors needed to compute the error
  zt_ij = v2t(z);  %3*3
  vt_i  = v2t(x1); %3*3
  vt_j  = v2t(x2);

   
  R_i  = vt_i(1:2,1:2); %2*2
  R_ij = zt_ij(1:2,1:2); %2*2
  ct_i = cos(x1(3));
  st_i = sin(x1(3));
  dRdT_i = [-st_i ct_i;-ct_i -st_i];
  
  % (TODO) compute the error
  e= t2v(invt(zt_ij)*(invt(vt_i)*vt_j));   
   


  % (TODO) compute the Jacobians of the error

  A = [ (-R_ij'*R_i') (R_ij'*dRdT_i*(x2(1:2)-x1(1:2))); 0 0 -1]; %3*3

  B = [R_ij'*R_i' [0  0]';0  0 1]; %3*3

end
