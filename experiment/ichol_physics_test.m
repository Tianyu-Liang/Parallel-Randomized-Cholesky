%H = load("../physics/parabolic_fem/parabolic_fem.mat");
%A = H.Problem.A;


mat_list = ["../physics/parabolic_fem/parabolic_fem-amd.mtx", "../physics/ecology2/ecology2-amd.mtx", ...
 "../physics/apache2/apache2-amd.mtx", "../physics/G3_circuit/G3_circuit-amd.mtx", "../physics/uniform_3D/uniform_3D-amd.mtx", ...
  "../physics/aniso_contrast_3D/aniso_contrast_3D-amd.mtx", "../physics/poisson_contrast_3D/poisson_contrast_3D-amd.mtx", "../physics/spe16m/spe16m-amd.mtx"];
write_list=["../physics/parabolic_fem/", "../physics/ecology2/", "../physics/apache2/", ...
 "../physics/G3_circuit/", "../physics/uniform_3D/", "../physics/aniso_contrast_3D/", "../physics/poisson_contrast_3D/", "../physics/spe16m/"];

drop_list = [4.5e-3, 2e-2, 2.2e-3, 4.5e-3, 5e-3, 2.499e-5, 1.45e-3, 6.3e-4];

% mat_list = ["../physics/poisson_contrast_3D/poisson_contrast_3D-amd.mtx"];
% write_list=["../physics/poisson_contrast_3D/"];

% drop_list = [1.45e-3];


for i = 1 : length(mat_list)

    % nnz_col = zeros(size(A, 1), 1);
    % for i = 1 : size(A, 1)
    %     nnz_col(i) = nnz(A(:, i));
    % end
    % [out, idx] = sort(nnz_col);
    % A = A(idx, idx);
    A = mmread(mat_list(i));
    A = A(1 : end - 1, 1 : end - 1);
    if mat_list(i) == "../physics/spe16m/spe16m-amd.mtx"
        A = -A;
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



% ichol_physics_test
% got here\n
% Elapsed time is 0.226167 seconds.
% problem nnz: 3674625, factor nnz: 4872874
% nnz ratio: 2.6522
% got here\n
% Elapsed time is 0.132170 seconds.
% problem nnz: 4995991, factor nnz: 6093451
% nnz ratio: 2.4393
% got here\n
% Elapsed time is 0.404889 seconds.
% problem nnz: 4817870, factor nnz: 7543801
% nnz ratio: 3.1316
% got here\n
% Elapsed time is 0.430360 seconds.
% problem nnz: 7660826, factor nnz: 11704604
% nnz ratio: 3.0557
% got here\n
% Elapsed time is 8.964151 seconds.
% problem nnz: 100088055, factor nnz: 181105813
% nnz ratio: 3.6189
% got here\n
% Elapsed time is 5.172410 seconds.
% problem nnz: 100088055, factor nnz: 130878371
% nnz ratio: 2.6153
% got here\n
% Elapsed time is 6.701597 seconds.
% problem nnz: 100088055, factor nnz: 160072777
% nnz ratio: 3.1986
% got here\n
% Elapsed time is 7.757611 seconds.
% problem nnz: 111640032, factor nnz: 170205187
% nnz ratio: 3.0492