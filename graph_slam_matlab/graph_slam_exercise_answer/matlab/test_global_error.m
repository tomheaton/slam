addpath('tools');

fid = fopen('test_global_error/test.g2o', 'w');
n = 15;
space_vec = zeros(n*3, 1);
edges = [];
for i=1:n
    x = rand(3,1);
    space_vec((i-1)*3+1:(i-1)*3+3) = x;
    fprintf(fid, 'VERTEX_SE2 %i %f %f %f\n', i, x(1), x(2), x(3));
end

for i=1:n

    id1 = randi([1, n])
    id2 = randi([1, n])
    if id1==id2
        continue
    end
    measurement = rand(3,1)
    ut = rand(6,1)
    information = [[ut(1) ut(2) ut(3)]; [ut(2) ut(4) ut(5)]; [ut(3) ut(5) ut(6)]]
    
    edges = [edges, struct('type', 'P', 'fromIdx', id1, 'toIdx', id2, 'measurement', measurement, 'information', information)]

    fprintf(fid, 'EDGE_SE2 %i %i %f %f %f %f %f %f %f %f %f\n', id1, id2, measurement(1), measurement(2), measurement(3), ut(1), ut(2), ut(3), ut(4), ut(5), ut(6));

end

g = struct('x', space_vec, 'edges', edges)

disp('global error:')
disp(compute_global_error(g))


% global error was: 10.2903
% we got 18.something
