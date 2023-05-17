% Compute the error of a pose-landmark constraint
% x 3x1 vector (x,y,theta) of the robot pose
% l 2x1 vector (x,y) of the landmark
% z 2x1 vector (x,y) of the measurement, the position of the landmark in
%   the coordinate frame of the robot given by the vector x
%
% Output
% e 2x1 error of the constraint
% A 2x3 Jacobian wrt x
% B 2x2 Jacobian wrt l
function [e, A, B] = linearize_pose_landmark_constraint(x, l, z)


  % (TODO) compute the error 
  xt = v2t(x);
  e = invt(xt) * [l;1] - [z;1];
  e = e(1:2);

  
  % (TODO) compute the Jacobians of the error
  Ri = xt(1:2,1:2);
  
  ct_i = cos(x(3));
  st_i = sin(x(3));
  dRdT_i = [-st_i ct_i;-ct_i -st_i];
 
  A = [ -Ri' dRdT_i*(l - x(1:2))];   %2*3
  B = Ri';  %2*2

end
