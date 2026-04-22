function [ err ] = mmwrite(filename,A,comment,field,precision)
    %
    % Function: mmwrite(filename,A,comment,field,precision)
    %
    %    Writes the sparse or dense matrix A to a Matrix Market (MM)
    %    formatted file.
    %
    % Required arguments:
    %
    %                 filename  -  destination file
    %
    %                 A         -  sparse or full matrix
    %
    % Optional arguments:
    %
    %                 comment   -  matrix of comments to prepend to
    %                              the MM file.  To build a comment matrix,
    %                              use str2mat. For example:
    %
    %                              comment = str2mat(' Comment 1' ,...
    %                                                ' Comment 2',...
    %                                                ' and so on.',...
    %                                                ' to attach a date:',...
    %                                                [' ',date]);
    %                              If ommitted, a single line date stamp comment
    %                              will be included.
    %
    %                 field     -  'real'
    %                              'complex'
    %                              'integer'
    %                              'pattern'
    %                              If ommitted, data will determine type.
    %
    %                 precision -  number of digits to display for real
    %                              or complex values
    %                              If ommitted, full working precision is used.
    %

    if ( nargin < 5 )
      precision = 16;
    end
    if ( nargin < 4 )
      field = '';
    end
    if ( nargin < 3 )
      comment = '';
    end

    mmfile = fopen(filename,'w');
    if ( mmfile == -1 )
     error('Cannot open file for output');
    end;

    [M,N] = size(A);

    %%%%%%%%%%%%%       This part for sparse matrices     %%%%%%%%%%%%%%%%
    if ( issparse(A) )

      [I,J,V] = find(A);

      % Determine field type
      if ( isempty(field) || ~strcmp(field,'pattern') )
        if ( any(imag(V)) )
          mattype = 'complex';
        else
          mattype = 'real';
        end
      else
        mattype = 'pattern';
      end

    %
    % Determine symmetry (vectorized):
    %
      if ( M ~= N )
        symm = 'general';
      else
        if ( nnz(A - A.') == 0 )
          symm = 'symmetric';
          ATEMP = tril(A);
          [I,J,V] = find(ATEMP);
        elseif ( nnz(A + A.') == 0 )
          symm = 'skew-symmetric';
          ATEMP = tril(A);
          [I,J,V] = find(ATEMP);
        elseif ( strcmp(mattype,'complex') && nnz(A - A') == 0 )
          symm = 'hermitian';
          ATEMP = tril(A);
          [I,J,V] = find(ATEMP);
        else
          symm = 'general';
        end
      end

      NZ = length(V);

    % Sparse coordinate format:

      rep = 'coordinate';

      fprintf(mmfile,'%%%%MatrixMarket matrix %s %s %s\n',rep,mattype,symm);
      [MC,~] = size(comment);
      if ( MC == 0 )
        fprintf(mmfile,'%% Generated %s\n',[date]);
      else
        for i=1:MC,
          fprintf(mmfile,'%%%s\n',comment(i,:));
        end
      end
      fprintf(mmfile,'%d %d %d\n',M,N,NZ);

      realformat = sprintf('%%d %%d %%.%dg\n',precision);
      cplxformat = sprintf('%%d %%d %%.%dg %%.%dg\n',precision,precision);

      if ( strcmp(mattype,'real') )
         fprintf(mmfile,realformat,[I J V]');
      elseif ( strcmp(mattype,'complex') )
         fprintf(mmfile,cplxformat,[I J real(V) imag(V)]');
      elseif ( strcmp(mattype,'pattern') )
         fprintf(mmfile,'%d %d\n',[I J]');
      else
         err = -1;
         disp('Unsupported mattype:')
         mattype
      end;

    %%%%%%%%%%%%%       This part for dense matrices      %%%%%%%%%%%%%%%%
    else
      % Determine field type
      if ( isempty(field) || ~strcmp(field,'pattern') )
        if ( any(imag(A(:))) )
          mattype = 'complex';
        else
          mattype = 'real';
        end
      else
        mattype = 'pattern';
      end

    %
    % Determine symmetry (vectorized):
    %
      if ( M ~= N )
        symm = 'general';
      else
        if ( isequal(A, A.') )
          symm = 'symmetric';
        elseif ( isequal(A, -A.') )
          symm = 'skew-symmetric';
        elseif ( strcmp(mattype,'complex') && isequal(A, A') )
          symm = 'hermitian';
        else
          symm = 'general';
        end
      end

    % Dense array format:

      rep = 'array';
      [MC,~] = size(comment);
      fprintf(mmfile,'%%%%MatrixMarket matrix %s %s %s\n',rep,mattype,symm);
      for i=1:MC,
        fprintf(mmfile,'%%%s\n',comment(i,:));
      end;
      fprintf(mmfile,'%d %d\n',M,N);

      realformat = sprintf('%%.%dg\n', precision);
      cplxformat = sprintf('%%.%dg %%.%dg\n', precision, precision);

      if ( strcmp(symm,'general') )
        coldata = A;
      else
        % Write only lower triangle for symmetric/hermitian/skew
        coldata = tril(A);
      end

      if ( strcmp(mattype,'real') )
         if ( strcmp(symm,'general') )
           fprintf(mmfile,realformat,A);
         else
           for j=1:N
             fprintf(mmfile,realformat,A(j:M,j));
           end
         end
      elseif ( strcmp(mattype,'complex') )
         if ( strcmp(symm,'general') )
           vals = [real(A(:)) imag(A(:))]';
           fprintf(mmfile,cplxformat,vals);
         else
           for j=1:N
             col = A(j:M,j);
             fprintf(mmfile,cplxformat,[real(col) imag(col)]');
           end
         end
      elseif ( strcmp(mattype,'pattern') )
         err = -2;
         disp('Pattern type inconsistent with dense matrix');
      else
         err = -2;
         disp('Unknown matrix type:');
         mattype
      end
    end

    fclose(mmfile);
