function  [A,rows,cols,entries,rep,field,symm] = mmread(filename)
    %
    % function  [A] = mmread(filename)
    %
    % function  [A,rows,cols,entries,rep,field,symm] = mmread(filename)
    %
    %      Reads the contents of the Matrix Market file 'filename'
    %      into the matrix 'A'.  'A' will be either sparse or full,
    %      depending on the Matrix Market format indicated by
    %      'coordinate' (coordinate sparse storage), or
    %      'array' (dense array storage).  The data will be duplicated
    %      as appropriate if symmetry is indicated in the header.
    %
    %      Optionally, size information about the matrix can be
    %      obtained by using the return values rows, cols, and
    %      entries, where entries is the number of nonzero entries
    %      in the final matrix. Type information can also be retrieved
    %      using the optional return values rep (representation), field,
    %      and symm (symmetry).
    %

    mmfile = fopen(filename,'r');
    if ( mmfile == -1 )
     disp(filename);
     error('File not found');
    end;

    header = fgets(mmfile);
    if (header == -1 )
      error('Empty file.')
    end

    [head0,header]   = strtok(header);
    [head1,header]   = strtok(header);
    [rep,header]     = strtok(header);
    [field,header]   = strtok(header);
    [symm,header]    = strtok(header);
    head1 = lower(head1);
    rep   = lower(rep);
    field = lower(field);
    symm  = lower(symm);
    if ( length(symm) == 0 )
       disp(['Not enough words in header line of file ',filename])
       disp('Recognized format: ')
       disp('%%MatrixMarket matrix representation field symmetry')
       error('Check header line.')
    end
    if ( ~ strcmp(head0,'%%MatrixMarket') )
       error('Not a valid MatrixMarket header.')
    end
    if (  ~ strcmp(head1,'matrix') )
       disp(['This seems to be a MatrixMarket ',head1,' file.']);
       disp('This function only knows how to read MatrixMarket matrix files.');
       disp('  ');
       error('  ');
    end

    % Read through comments, ignoring them

    commentline = fgets(mmfile);
    while length(commentline) > 0 & commentline(1) == '%',
      commentline = fgets(mmfile);
    end

    % Read size information, then branch according to
    % sparse or dense format

    if ( strcmp(rep,'coordinate')) %  read matrix given in sparse
                                  %  coordinate matrix format

      [sizeinfo,count] = sscanf(commentline,'%d%d%d');
      while ( count == 0 )
         commentline =  fgets(mmfile);
         if (commentline == -1 )
           error('End-of-file reached before size information was found.')
         end
         [sizeinfo,count] = sscanf(commentline,'%d%d%d');
         if ( count > 0 & count ~= 3 )
           error('Invalid size specification line.')
         end
      end
      rows = sizeinfo(1);
      cols = sizeinfo(2);
      entries = sizeinfo(3);

      if  ( strcmp(field,'real') )               % real valued entries:

        T = fscanf(mmfile,'%f');
        if ( numel(T) ~= 3*entries )
           error('Data file does not contain expected amount of data. Check that number of data lines matches nonzero count.');
        end
        T = reshape(T,3,entries)';
        I = T(:,1);
        J = T(:,2);
        V = T(:,3);
        if ( strcmp(symm,'symmetric') || strcmp(symm,'hermitian') )
          off = (I ~= J);
          I = [I; J(off)];
          J = [J; T(off,1)];
          V = [V; V(off)];
          A = sparse(I, J, V, rows, cols);
          entries = nnz(A);
        elseif ( strcmp(symm,'skew-symmetric') )
          off = (I ~= J);
          offI = T(off,1); offJ = T(off,2); offV = T(off,3);
          I = [T(:,1); offJ];
          J = [T(:,2); offI];
          V = [T(:,3); -offV];
          A = sparse(I, J, V, rows, cols);
          entries = nnz(A);
        else
          A = sparse(I, J, V, rows, cols);
        end

      elseif   ( strcmp(field,'complex'))            % complex valued entries:

        T = fscanf(mmfile,'%f');
        if ( numel(T) ~= 4*entries )
           error('Data file does not contain expected amount of data. Check that number of data lines matches nonzero count.');
        end
        T = reshape(T,4,entries)';
        I = T(:,1);
        J = T(:,2);
        V = T(:,3) + T(:,4)*1i;
        if ( strcmp(symm,'symmetric') )
          off = (I ~= J);
          I = [I; J(off)];
          J = [T(:,2); T(off,1)];
          V = [V; V(off)];
          A = sparse(I, J, V, rows, cols);
          entries = nnz(A);
        elseif ( strcmp(symm,'hermitian') )
          off = (I ~= J);
          I = [I; J(off)];
          J = [T(:,2); T(off,1)];
          V = [V; conj(V(off))];
          A = sparse(I, J, V, rows, cols);
          entries = nnz(A);
        elseif ( strcmp(symm,'skew-symmetric') )
          off = (I ~= J);
          I = [I; J(off)];
          J = [T(:,2); T(off,1)];
          V = [V; -V(off)];
          A = sparse(I, J, V, rows, cols);
          entries = nnz(A);
        else
          A = sparse(I, J, V, rows, cols);
        end

      elseif  ( strcmp(field,'pattern'))    % pattern matrix (no values given):

        T = fscanf(mmfile,'%f');
        if ( numel(T) ~= 2*entries )
           error('Data file does not contain expected amount of data. Check that number of data lines matches nonzero count.');
        end
        T = reshape(T,2,entries)';
        I = T(:,1);
        J = T(:,2);
        V = ones(entries,1);
        if ( strcmp(symm,'symmetric') || strcmp(symm,'hermitian') )
          off = (I ~= J);
          I = [I; J(off)];
          J = [T(:,2); T(off,1)];
          V = [V; ones(sum(off),1)];
          A = sparse(I, J, V, rows, cols);
          entries = nnz(A);
        elseif ( strcmp(symm,'skew-symmetric') )
          off = (I ~= J);
          I = [I; J(off)];
          J = [T(:,2); T(off,1)];
          V = [V; -ones(sum(off),1)];
          A = sparse(I, J, V, rows, cols);
          entries = nnz(A);
        else
          A = sparse(I, J, V, rows, cols);
        end

      end

    elseif ( strcmp(rep,'array') ) %  read matrix given in dense
                                   %  array (column major) format

      [sizeinfo,count] = sscanf(commentline,'%d%d');
      while ( count == 0 )
         commentline =  fgets(mmfile);
         if (commentline == -1 )
           error('End-of-file reached before size information was found.')
         end
         [sizeinfo,count] = sscanf(commentline,'%d%d');
         if ( count > 0 & count ~= 2 )
           error('Invalid size specification line.')
         end
      end
      rows = sizeinfo(1);
      cols = sizeinfo(2);
      entries = rows*cols;
      if  ( strcmp(field,'real') )               % real valued entries:
        A = fscanf(mmfile,'%f');
        if ( strcmp(symm,'symmetric') | strcmp(symm,'hermitian') | strcmp(symm,'skew-symmetric') )
          for j=1:cols-1,
            currenti = j*rows;
            A = [A(1:currenti); zeros(j,1);A(currenti+1:length(A))];
          end
        elseif ( ~ strcmp(symm,'general') )
          error('Unrecognized symmetry. Check header.');
        end
          A = reshape(A,rows,cols);
      elseif  ( strcmp(field,'complex'))         % complex valued entries:
        T = fscanf(mmfile,'%f');
        T = reshape(T, 2, [])';
        A = T(:,1) + T(:,2)*1i;
        if ( strcmp(symm,'symmetric') | strcmp(symm,'hermitian') | strcmp(symm,'skew-symmetric') )
          for j=1:cols-1,
            currenti = j*rows;
            A = [A(1:currenti); zeros(j,1);A(currenti+1:length(A))];
          end
        elseif ( ~ strcmp(symm,'general') )
          error('Unrecognized symmetry. Check header.');
        end
        A = reshape(A,rows,cols);
      elseif  ( strcmp(field,'pattern'))    % pattern (makes no sense for dense)
       error('Pattern matrix type invalid for array storage format.');
      else                                 % Unknown matrix type
       error('Invalid matrix type specification. Check header against MM documentation.');
      end

      % Symmetry expansion for dense matrices
      if ( strcmp(symm,'symmetric') )
         A = A + A.' - diag(diag(A));
         entries = nnz(A);
      elseif ( strcmp(symm,'hermitian') )
         A = A + A' - diag(diag(A));
         entries = nnz(A);
      elseif ( strcmp(symm,'skew-symmetric') )
         A = A - A';
         entries = nnz(A);
      end
    end

    fclose(mmfile);
