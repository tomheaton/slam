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