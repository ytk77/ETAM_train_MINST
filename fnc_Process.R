fnc_hardlim <- function(x)
{
    x = 1*(x>0)
    x[ x==0 ] = -1
    return(x)
}