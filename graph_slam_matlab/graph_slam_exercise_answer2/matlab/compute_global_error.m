% Computes the total error of the graph
function Fx = compute_global_error(g)

Fx = 0;

% Loop over all edges
for eid = 1:length(g.edges)
  edge = g.edges(eid);

  % pose-pose constraint
  if (strcmp(edge.type, 'P') ~= 0)

    x1 = g.x(edge.fromIdx:edge.fromIdx+2);  % the first robot pose
    x2 = g.x(edge.toIdx:edge.toIdx+2);      % the second robot pose
    z12  = edge.measurement;                % the measurement 
    info12  = edge.information;             % the information matrix corresponding to the measurement 

    % (TODO) compute the error of the constraint and add it to Fx.
    es= t2v(invt(v2t(z12))*(invt(v2t(x1))*v2t(x2)));   
    Fx = Fx + es'*info12*es;

  % pose-landmark constraint
  elseif (strcmp(edge.type, 'L') ~= 0)
    x = g.x(edge.fromIdx:edge.fromIdx+2);  % the robot pose
    l = g.x(edge.toIdx:edge.toIdx+1);      % the landmark
    z  = edge.measurement;                 % the measurement 
    info  = edge.information;              % the information matrix corresponding to the measurement 

    % (TODO) compute the error of the constraint and add it to Fx.
    es = invt(v2t(x))*[l;1] - [z;1];
    es = es(1:2);
    Fx = Fx + es' * info * es;

  end
end
