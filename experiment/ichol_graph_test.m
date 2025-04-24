%H = load("../physics/parabolic_fem/parabolic_fem.mat");
%A = H.Problem.A;


mat_list = [ "../physics/ecology1/ecology1-amd.mtx", "../data/GAP-road/GAP-road-amd.mtx", "../data/com-LiveJournal/com-LiveJournal-amd.mtx", ... 
    "../data/europe_osm/europe_osm-amd.mtx", "../data/delaunay_n24/delaunay_n24-amd.mtx", ... 
    "../data/venturiLevel3/venturiLevel3-amd.mtx", "../data/belgium_osm/belgium_osm-amd.mtx"];

write_list = ["../physics/ecology1/", "../data/GAP-road/", "../data/com-LiveJournal/", ... 
        "../data/europe_osm/", "../data/delaunay_n24/", ... 
        "../data/venturiLevel3/", "../data/belgium_osm/"];

drop_list = [2e-2, 3e-3, 1.2e-4, 1e-3, 1.2e-2, 1.9e-2, 3e-3];


for i = 1 : length(mat_list)

    % nnz_col = zeros(size(A, 1), 1);
    % for i = 1 : size(A, 1)
    %     nnz_col(i) = nnz(A(:, i));
    % end
    % [out, idx] = sort(nnz_col);
    % A = A(idx, idx);
    A = mmread(mat_list(i));
    n = size(A, 1);
    if A(1, 1) == 0
        A = -abs(A);
        A = A - diag(A);
        A = A - sparse(1 : n, 1 : n, sum(A), n, n);
    end
    

    disp("got here\n");

    alpha = 1e-6;
    tic;
    %L1 = ichol(A, struct('type','ict','droptol',1e-2,'diagcomp',alpha));
    L1 = ichol(A, struct('type','ict','droptol', drop_list(i)));
    toc;
    disp("problem nnz: " + nnz(A) +  ", factor nnz: " + nnz(L1));
    disp("nnz ratio: " + 2 * nnz(L1) / nnz(A));
    
    %mmwrite(write_list(i) + "_ic_amd.mtx", L1);
    %b = A * rand(size(A, 1), 1);
    %[x1,fl1,rr1,it1,rv1] = pcg(A,b,1e-6,100,L1,L1');
   % x = pcg(A,b,1e-6,1000,L1,L1');
    %disp(norm(A * x - b) / norm(b))

end

% got here\n
% Elapsed time is 0.127277 seconds.
% problem nnz: 4995991, factor nnz: 6093533
% nnz ratio: 2.4394
% got here\n
% Elapsed time is 1.675887 seconds.
% problem nnz: 81655971, factor nnz: 78806316
% nnz ratio: 1.9302
% got here\n
% Elapsed time is 193.477825 seconds.
% problem nnz: 73360340, factor nnz: 243472403
% nnz ratio: 6.6377
% got here\n
% Elapsed time is 2.702340 seconds.
% problem nnz: 159021338, factor nnz: 151167124
% nnz ratio: 1.9012
% got here\n
% Elapsed time is 5.813855 seconds.
% problem nnz: 117440418, factor nnz: 116656558
% nnz ratio: 1.9867
% got here\n
% Elapsed time is 0.898312 seconds.
% problem nnz: 20135293, factor nnz: 24446806
% nnz ratio: 2.4283
% got here\n
% Elapsed time is 0.076615 seconds.
% problem nnz: 4541235, factor nnz: 4417866
% nnz ratio: 1.9457