fid = fopen('test/invt/input_test.txt', 'w');
fid2 = fopen('test/invt/result_test.txt', 'w');
for i=1:5000
    x = rand(3,3);
    fprintf(fid, '%f %f %f\n', x(1,1), x(1,2), x(1,3));
    fprintf(fid, '%f %f %f\n', x(2,1), x(2,2), x(2,3));
    fprintf(fid, '%f %f %f\n', x(3,1), x(3,2), x(3,3));
    fprintf(fid, '\n');

    result = invt(x);
    %fprintf(fid2, '%f %f %f\n', result(1), result(2), result(3));
    fprintf(fid2, '%f %f %f\n', result(1,1), result(1,2), result(1,3));
    fprintf(fid2, '%f %f %f\n', result(2,1), result(2,2), result(2,3));
    fprintf(fid2, '%f %f %f\n', result(3,1), result(3,2), result(3,3));
end

data = load('../../data/intel.mat');
dataFile = fopen('../../data/intel.mat', 'r');
outFile = fopen('test/intel_out.txt', 'w');

fprintf(outFile, '%f', dataFile);
