more off;
clear all;
close all;

addpath('tools');

% load the graph into the variable g
% only leave one line uncommented

% simulation datasets
%load ../data/simulation-pose-pose.mat
%load ../data/simulation-pose-landmark.mat

% real-world datasets
load ../data/intel.mat
%load ../data/dlr.mat

% g = read_graph('../data/intel_copy.g2o');

% create intel.g2o from intel.mat
outFile = fopen('intel.g2o', 'w');

vertexId = 0;

for n = 1:3:(length(g.x) - 3)
    fprintf( ...
        outFile, ...
        'VERTEX_SE2 %d %f %f %f\n', ...
        vertexId, g.x(n), g.x(n+1), g.x(n+2) ...
        );
    vertexId = vertexId + 1;
end

disp("done vertices");

for n = 1:length(g.edges)
    data = g.edges(n);
    m = data.measurement;
    i = data.information;
    fprintf( ...
        outFile, ...
        'EDGE_SE2 %d %d %f %f %f %f %f %f %f %f %f\n', ...
        data.to, data.from, m(1), m(2), m(3), i(1,1), i(1,2), i(1,3), i(2,2), i(2,3), i(3,3) ...
        );
end

disp("done edges");

disp("end here");

% the number of iterations
numIterations = 100;

% maximum allowed dx
EPSILON = 10^-4;

% Error
err = 0;

% plot the initial state of the graph
plot_graph(g, 0);

fprintf('Initial error %f\n', compute_global_error(g));

% iterate
for i = 1:numIterations 
  fprintf('Performing iteration %d\n', i);

  % compute the incremental update dx of the state vector
  dx = linearize_and_solve(g);

  % (TODO) apply the solution to the state vector g.x

    g.x= g.x + dx;

  % plot the current state of the graph
  plot_graph(g, i);

  % Compute the global error given the current graph configuration
  err = compute_global_error(g);

  % Print current error
  fprintf('Current error %f\n', err);

  % termination criterion  
  if(max(abs(dx)) < EPSILON)
	disp('Converged!!');
	break;
  end

end

fprintf('Done!\nFinal error %f\n', err);
