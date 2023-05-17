% performs one iteration of the Gauss-Newton algorithm
% each constraint is linearized and added to the Hessian

function dx = linearize_and_solve(g)

nnz = nnz_of_graph(g);

% allocate the sparse H and the vector b
H = spalloc(length(g.x), length(g.x), nnz);
b = zeros(length(g.x), 1);

needToAddPrior = true;

% compute the addend term to H and b for each of our constraints
disp('linearize and build system');
for eid = 1:length(g.edges)
  edge = g.edges(eid);

  % pose-pose constraint
  if (strcmp(edge.type, 'P') ~= 0)
    % edge.fromIdx and edge.toIdx describe the location of
    % the first element of the pose in the state vector
    % You should use also this index when updating the elements
    % of the H matrix and the vector b.
    % edge.measurement is the measurement
    % edge.information is the information matrix
    x1 = g.x(edge.fromIdx:edge.fromIdx+2);  % the first robot pose
    x2 = g.x(edge.toIdx:edge.toIdx+2);      % the second robot pose

    % Computing the error and the Jacobians
    % e the error vector
    % A Jacobian wrt x1
    % B Jacobian wrt x2
    [e, A, B] = linearize_pose_pose_constraint(x1, x2, edge.measurement);


    % (TODO) compute and add the term introduced by the pose-pose constraint to H and b

      b_i  = -(e' * edge.information * A)';
      b_j  = -(e' * edge.information * B)';
      H_ii = A' * edge.information * A;
      H_ij = A' * edge.information * B ;
      H_jj = B' * edge.information * B ;


    % adding them to the matrix and b
    
    H(edge.fromIdx:edge.fromIdx+2, edge.fromIdx:edge.fromIdx+2) = H(edge.fromIdx:edge.fromIdx+2, edge.fromIdx:edge.fromIdx+2) +  H_ii;
    H(edge.fromIdx:edge.fromIdx+2, edge.toIdx:edge.toIdx+2) = H(edge.fromIdx:edge.fromIdx+2, edge.toIdx:edge.toIdx+2) + H_ij;
    H(edge.toIdx:edge.toIdx+2, edge.fromIdx:edge.fromIdx+2) = H(edge.toIdx:edge.toIdx+2, edge.fromIdx:edge.fromIdx+2) + H_ij';
    H(edge.toIdx:edge.toIdx+2, edge.toIdx:edge.toIdx+2) = H(edge.toIdx:edge.toIdx+2, edge.toIdx:edge.toIdx+2) + H_jj;

    b(edge.fromIdx:edge.fromIdx+2) = b(edge.fromIdx:edge.fromIdx+2) + b_i;
    b(edge.toIdx:edge.toIdx+2) = b(edge.toIdx:edge.toIdx+2) + b_j;

    if (needToAddPrior)
      % (no need to do, already given) TODO: add the prior for one pose of this edge
      % This fixes one node to remain at its current location
      H(edge.fromIdx:edge.fromIdx+2, edge.fromIdx:edge.fromIdx+2) = H(edge.fromIdx:edge.fromIdx+2, edge.fromIdx:edge.fromIdx+2) + 1000*eye(3);
      % if want this node at the origin
      % b(edge.fromIdx:edge.fromIdx+2) = b(edge.fromIdx:edge.fromIdx+2)-(x1'*1000*eye(3))'; 
      
      needToAddPrior = false;
    end

  % pose-landmark constraint
  elseif (strcmp(edge.type, 'L') ~= 0)
    % edge.fromIdx and edge.toIdx describe the location of
    % the first element of the pose and the landmark in the state vector
    % You should use also this index when updating the elements
    % of the H matrix and the vector b.
    % edge.measurement is the measurement
    % edge.information is the information matrix
    x1 = g.x(edge.fromIdx:edge.fromIdx+2);  % the robot pose
    x2 = g.x(edge.toIdx:edge.toIdx+1);      % the landmark

    % Computing the error and the Jacobians
    % e the error vector
    % A Jacobian wrt x1
    % B Jacobian wrt x2
    [e, A, B] = linearize_pose_landmark_constraint(x1, x2, edge.measurement);


    % (TODO) compute and add the term introduced by the pose-landmark constraint to H and b


%     TODO: YOU_NEED_TO_ADD_YOUR_CODE_HERE
      b_i  = -(e' * edge.information * A)';
      b_j  = -(e' * edge.information * B)';
      H_ii = A' * edge.information * A;
      H_ij = A' * edge.information * B ;
      H_jj = B' * edge.information * B ;
 

    % adding them to the matrix and b
    H(edge.fromIdx:edge.fromIdx+2, edge.fromIdx:edge.fromIdx+2) = H(edge.fromIdx:edge.fromIdx+2, edge.fromIdx:edge.fromIdx+2) + H_ii;
    H(edge.fromIdx:edge.fromIdx+2, edge.toIdx:edge.toIdx+1) = H(edge.fromIdx:edge.fromIdx+2, edge.toIdx:edge.toIdx+1) + H_ij;
    H(edge.toIdx:edge.toIdx+1, edge.fromIdx:edge.fromIdx+2) = H(edge.toIdx:edge.toIdx+1, edge.fromIdx:edge.fromIdx+2) + H_ij';
    H(edge.toIdx:edge.toIdx+1, edge.toIdx:edge.toIdx+1) = H(edge.toIdx:edge.toIdx+1, edge.toIdx:edge.toIdx+1) + H_jj;

    b(edge.fromIdx:edge.fromIdx+2) = b(edge.fromIdx:edge.fromIdx+2) + b_i;
    b(edge.toIdx:edge.toIdx+1) = b(edge.toIdx:edge.toIdx+1) + b_j;

  end
end

disp('solving system');

% Solve the linear system, whereas the solution should be stored in dx
% Remember to use the backslash operator instead of inverting H

%  solve the linear system
dx = H\b;

end
