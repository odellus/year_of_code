#include "math.h"

#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#define max(X, Y) ((X) > (Y) ? (X) : (Y))

int mindex3(int x, int y, int z, int sizx, int sizy, int sizz, int wrap)
{
    int index;
    if (wrap == 1)
    {
        x = (x % sizx + sizx) % sizx;
        y = (y % sizy + sizy) % sizy;
        z = (z % sizz + sizz) % sizz;
    }
    else if (wrap>1)
    {
        x = max(x,0);
        y = max(y,0);
        z = max(z,0);

        x = min(x, sizx-1);
        y = min(y, sizy-1);
        z = min(z, sizz-1)
    }
    index = z * sizx * sizy + y * sizx + x;
    return index;
}

PyObject *_wrap
