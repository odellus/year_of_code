void inplace(double *invec, double *outvec, int n, int m) {
    int i;
    for (i=0; i < n; i++) {
        outvec[i] = 2*invec[i];
    }
}
