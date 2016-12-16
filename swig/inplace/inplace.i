%module inplace

%{
    #define SWIG_FILE_WITH_INIT
    #include "inplace.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double* INPLACE_ARRAY1, double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* invec, double* outvec, int n, int m)}
%include "inplace.h"
