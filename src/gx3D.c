/* 3D version of the code (TR) modified 2023
 
// to be compiled with
// gcc -DNABLA -DLMAX=40 -O3 -lm -lgsl -lgslcblas -o gx3D gx3D.c

//On ubuntu compilation order matters: use on Ubuntu: gcc -DNABLA -DLMAX=40 -O3 -o gx3D gx3D.c -lgsl -lgslcblas -lm

//to be run with 
// ./gx3D lambda=0.71 sigma=0 sdev=0.05747 alpha=.2 blocks=35 seed=1234 > gtdati &


/* code for solving the beta-scaling equation of MCT,
   calculating an average over many distance parameters,
   and taking into account a gradient-square coupling between them
   following an idea by Tommaso Rizzo, arXiv:1307.4303 (2013)
   Thomas Voigtmann (tv) 2013-07-29, 2013-08-08 */
/* the equation this code tries to solve, is

                    d^2                             d  /t
   sigma(x) + alpha ---- g(x,t) + lambda g(x,t)^2 = -- |  g(x,t-t')g(x,t') dt'
                    dx^2                            dt /0

   (for the time being, x is a one-dimensional spatial variable)

   the code with NABLA un-defined at compile time corresponds to a collection of 
   uncorrelated systems.
 
*/

/* compile using
   gcc -DHG_VERSION="`hg identify -n`" -O3 -lm -lgsl -lgslcblas -o gx gx.c
   this requires the GSL (GNU scientific library) for a random-number generator
   omit the HG_VERSION bit if you don't run mercurial for version control
   then run the code like
   ./gx lambda=0.75 sigma=0.1 var=0.1 >/tmp/gt.dat
   you must add -DNABLA to switch on to the code with spatial coupling
   you must add -DNMAX=100 or similar to change the number of beta correlators

   parameters to be set on the command line:
   h		initial step size			default 1e-5
   blocksize	size of h=const blocks			default 256
   blocks       number of blocks with constant h	default 40
   maxinit	length of short-time initialization	default 80
   maxiter	max.iterations for root finding		default 1000
   accuracy	accuracy goal for iterations		default 1e-9
   filter	output only every filter-th grid point	default 10
   lambda	MCT exponent parameter			default 0.75
   sigma	MCT distance parameter (average)	default 0
   sdev		standard deviation of sigma			default 0
   t0		initial time scale			default 1.0
   alpha	scale factor for dx			default 1.0
   seed		seed for random-number generator	default 0
*/

/*
Modified by TR 12-11-2013 in order to converge at alpha=1.
 
 alpha=1 seems to be able to give sufficiently smooth curves.
 (1) the solution is initialized by extrapolation from values at lower times because in presence of a strong gradient the problem locally looks much more homogeneous
 (2) the blocksize needs to be increased, for lambda=.75 alpha=1, and var=.1 blocksize=4096 is ok
 for sigma=0 and sigma=-.5.
 (3) the line 260 has been uncommented // now is no longer line 260, it is 320

 Further modification 2018
 
 (4)  I have implemented the corrections to the initial conditions in order to have the correct behavior
 for the fluctuations. Thus g[n*blocksize+i] = pow(t/t0,-a)+A1*pow(t/t0,a)*sigma[n]; (line 199) See further comments in the code.
 

 Further modifications 2019
 (5) I have commented a part that imposed that the sum of the random fields should be zero. (line 475)
 
 modification 2023
 (6) I have corrected a part that fixed sigma to zero when I modified the code in 2019
 
 
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef double fp;
#include <gsl/rng/gsl_rng.h>
#include <gsl/randist/gsl_randist.h>

// the number of correlators to average over is hard-coded here!
// it can be changed at compile-time
#if !defined(LMAX)
#define LMAX 10
#endif
// by default, do not include gradient coupling term
//#define NABLA

/* here are the parameters to be set from the command line:
   lambda = exponent parameter
   sigma = array of distance parameters
   t0 = initial time scale (can be set to unity) */
uint NMAX=LMAX*LMAX*LMAX;
uint L2=LMAX*LMAX;
fp lambda, sigma[LMAX*LMAX*LMAX], t0;
fp a; // MCT exponent a, will be calculated from lambda
/* we include an adjustable scale parameter for the spatial length */

fp A1; // MCT factor to compute the susceptibility
fp alpha;

/* parameters for the numerics:
   h = initial step size dt
   blocksize = number of steps to calculate with a single step size h
   blocks = number of blocks to calculate with increasing h
   maxiter = maximum number of iterations for iterative root finding
   accuracy = desired numerical accuracy */
fp h;
uint blocksize;
uint blocks;
uint maxiter;
fp accuracy;
uint halfblocksize; // will be calculated

/* set filter to a nonzero value to reduce output */
uint filter;

/* arrays to store the solutions:
   g[i] will be g(t) at time h*iN
   dG[i] will be (1/h)\int g(s)ds where the integral is from (i-1)*h to i*h */
fp *g, *dG;

/* regula falsi method to determine the root of a given function
   starts with two initial guesses x0 and x1 which ideally should bracket
   a single root of f(x) */
fp regula_falsi (fp x0, fp x1, fp (*f)()) {
  fp xguess, fguess, delta_x=0;
  fp xa, xb, fa, fb;
  uint num_iter=0;
  if (x0<x1) {
    xa=x0; xb=x1; fa=f(x0); fb=f(x1);
  } else if (x0>x1) {
    xa=x1; xb=x0; fa=f(x1); fb=f(x0);
  } else { // x0==x1?
    xa=x0; xb=x1+1.1*accuracy; fa=f(xa); fb=f(xb);
  }
  while (!num_iter ||
         ((delta_x = xb-xa) > accuracy && num_iter<=maxiter)) {
    // basic idea of regula falsi: linear approximation of f(x)
    // y=mx+c with c=f(a)-m*a, and m=(f(b)-f(a))/(b-a) (secant method)
    xguess = xa - (delta_x / (fb-fa)) * fa;
    fguess = f(xguess);
    // now, we have to find out, whether we guessed too high or too low
    // The first to if-cases handle a case that is not supposed to happen
    // for the "original" regula falsi (no root within [a,b]). Convergence
    // cannot be proven then, but for "well-behaved" functions, we can
    // continue by extending the interval again...
    if (xguess<xa) { xb=xa; xa=xguess; fb=fa; fa=fguess; }
    else if (xguess>xb) { xa=xb; xb=xguess; fa=fb; fb=fguess; }
    else if ((fguess>0 && fa<0) || (fguess<0 && fa>0)) {
      // f(xguess) and f(a) have opposite signs
      // then there must be a root in [a,xguess] (suppose f(x) is continuous)
      xb = xguess;
      fb = fguess;
    } else {
      // same argument for f(b) and f(xguess)
      xa = xguess;
      fa = fguess;
    }
    if (fguess==0 || (fguess>0 && fguess<accuracy)
                  || (fguess<0 && fguess>accuracy))
      break;
    num_iter++;
  }
  return xguess;
}

/* MCT exponent function
   the roots of f(x) = Gamma(1-x)^2/Gamma(1-2x) - lambda
   are the MCT exponents a,-b */
fp exponent_func (fp x) {
  fp xc=1-x; fp g=tgamma(xc); return (g*(g/tgamma(xc-x))-lambda);
}

/* initialize the solution scheme
   the first imax values of the first block are pre-calculated using the
   known asymptotic short-time behavior, g(t)~(t/t0)^-a 
 IMPORTANT modification I have added also the correction due to the random temperature because otherwise 
 the fluctuations were not ok at short times.
 */
void initial_values (uint imax) {
  uint i,n;
  fp t;
  for (n=0; n<NMAX; ++n) {
    g[n*blocksize+0]=0;
    for (i=1; i<imax; i++) {
      t = i*h;
      g[n*blocksize+i] = pow(t/t0,-a)+A1*pow(t/t0,a)*sigma[n];
    }
    dG[n*blocksize+1] = pow(h/t0,-a)/(1-a)+A1*sigma[n]*pow(h/t0,a)/(1+a);
    for (i=2; i<=imax; i++) {
      dG[n*blocksize+i] = pow(h/t0,-a)/(1.-a)
        * (pow((1./i),a-1) - pow(i-1,1.-a))+A1*sigma[n]*pow(h/t0,a)/(1.+a)
        * (pow((1./i),-a-1) - pow(i-1,1.+a));
    }
  }
}

/* "decimization": transfer a solution known on one block (step size h)
   to half a block with double the step size */
void decimize () {
  uint i, doublei, n;

  for (n=0; n<NMAX; ++n) {
    for (i=1; i<(halfblocksize/2); i++) {
      doublei = i+i;
      dG[n*blocksize+i]
        = 0.5 * (dG[n*blocksize+doublei-1] + dG[n*blocksize+doublei]);
    }
    for (i=halfblocksize/2; i<halfblocksize; i++) {
      doublei = i+i;
      dG[n*blocksize+i] = 0.25 * (g[n*blocksize+doublei-2]
                        + 2*g[n*blocksize+doublei-1] + g[n*blocksize+doublei]);
    }
    dG[n*blocksize+halfblocksize] = g[n*blocksize+blocksize-1];
    for (i=0; i<halfblocksize; i++) {
      doublei = i+i;
      g[n*blocksize+i] = g[n*blocksize+doublei];
    }
  }
  h = h*2.0;
}

// helper function to find the root of (2*_b*x - lambda*x*x + _a)
fp _a, _b;
fp glx (fp x) {
  return 2*_b * x - lambda*x*x + _a;
}

/* main part of the numerical procedure: solution on equidistant time-grid */
void solve_block (uint istart, uint iend) {
  uint i, ibar, itmp, itmp1, k, n, ix, iy, iz;
  uint iterations, passed;
  fp C[NMAX], newg[NMAX];
  uint nt,six,sixl,sixr,siy,siyl,siyr,siz,sizl,sizr;
  fp cn, dGn;

  /* for each time step i ... */
  for (i=istart; i<iend; i++) {
    for (n=0; n<NMAX; ++n) {
      ibar = i/2;
      itmp = i-1;
      itmp1 = i;
      /* approximate the convolution integral */
      C[n] = g[n*blocksize+i-ibar]*g[n*blocksize+ibar]
           - 2*g[n*blocksize+i-1]*dG[n*blocksize+1];
      for (k=2; k<=ibar; k++) {
        itmp--;  /* itmp is  i-k   */
        itmp1--; /* itmp1 is i-k+1 */
        C[n] += 2*(g[n*blocksize+itmp1]-g[n*blocksize+itmp])*dG[n*blocksize+k];
      }
      if (i-ibar>ibar) {
        itmp--;
        itmp1--;
        C[n] += (g[n*blocksize+itmp1] - g[n*blocksize+itmp])*dG[n*blocksize+k];
      }
      C[n] -= sigma[n];
    }

    // this is the solution of the equation without the gradient coupling
    for (n=0; n<NMAX; ++n) {
      dGn = dG[n*blocksize+1]/lambda;
      g[n*blocksize+i] = dGn - sqrt(dGn*dGn + C[n]/lambda );
    }
    #ifdef NABLA
      // if the gradient-square term is there, we solve the implicit equation
      // by iteration, using the above no-coupling solution as a start
      //Here I want to modify by initializing by extrapolation, because if alpha is large the solution may be quite distant from the zero coupling solution. 
      
  //extrapolation as a start    
      for (n=0; n<NMAX; ++n) {
      g[n*blocksize+i] =2 *g[n*blocksize+i-1] - g[n*blocksize+i-2] ;
    }
      
      for (n=0; n<NMAX; ++n) newg[n]=g[n*blocksize+i];
      iterations = 0;
      do {
        passed=1;
	
       
	//this is where there is the difference with the 1D case:
         for (ix=0; ix<LMAX; ++ix) {
	    sixl = (ix>0 ? ix-1 : LMAX-1) * L2;//li devi definire
            sixr = (ix<LMAX-1 ? ix+1 : 0) * L2;
	    six  = ix * L2;  
	  for (iy=0; iy<LMAX; ++iy) {
	    siyl = (iy>0 ? iy-1 : LMAX-1) * LMAX;//li devi definire
            siyr = (iy<LMAX-1 ? iy+1 : 0) * LMAX;
	    siy= iy *LMAX;
	    for (iz=0; iz<LMAX; ++iz) {
	      sizl = (iz>0 ? iz-1 : LMAX-1);//li devi definire
              sizr = (iz<LMAX-1 ? iz+1 : 0);
	      siz=iz;  
	      
	      nt=six+siy+siz;
	      
          _a = C[nt] - (g[(sixl+siy+siz)*blocksize+i] +g[(sixr+siy+siz)*blocksize+i] +g[(six+siyl+siz)*blocksize+i] +g[(six+siyr+siz)*blocksize+i] +g[(six+siy+sizl)*blocksize+i] +g[(six+siy+sizr)*blocksize+i] )*alpha;
	  
	  _b = dG[nt*blocksize+1] + 3.*alpha;
	  newg[nt] = regula_falsi (g[nt*blocksize+i],newg[nt],glx);
          if (fabs(newg[nt]-g[nt*blocksize+i])<fabs(accuracy*g[nt*blocksize+i])
            || iterations>maxiter) {} else passed=0;
        
	  
	    }
	  }
	  
	}
        
        
        
        for (n=0; n<NMAX; ++n) g[n*blocksize+i] = newg[n];// I have uncommented this line, why is it commented? It is not solving the actual
        // equation and most probably exits only at maxiter
        if (passed) {
          for (n=0; n<NMAX; ++n) g[n*blocksize+i] = newg[n];
          break;
        }
        ++iterations;
      } while (1);
    #endif
    if (i<=halfblocksize) {
      for (n=0; n<NMAX; ++n)
        dG[n*blocksize+i] = 0.5 * (g[n*blocksize+i-1] + g[n*blocksize+i]);
    }
  }
}
 
/* output calculated values
   first two columns will be t and <g_n(t)>, after that all individual
   g_n(t) follow */
void output (uint istart, uint iend) {
  uint i,n;
  fp gavg;
  for (i=istart; i<iend; i++) {
    if (filter && i%filter)
      continue; /* skip all but every filter for output to save disk space */
    gavg=0; for (n=0; n<NMAX; ++n) { gavg+=g[n*blocksize+i]; } gavg/=NMAX;
    printf ("%.15le %.15le", i*h, gavg);
        for (n=0; n<NMAX; ++n) { printf (" %.15le",g[n*blocksize+i]); }
    printf ("\n");
    
  }
  fflush(stdout);
    }
/* output calculated values
   first two columns will be t and <g_n(t)>, after that all individual
   g_n(t) follow */
void outputcorr (uint istart, uint iend) {
  uint i,n;
  fp gavg;
  fp g2;
  uint r,ix,iy,iz;
  for (i=istart; i<iend; i++) {
    if (filter && i%filter)
      continue; /* skip all but every filter for output to save disk space */
    gavg=0; for (n=0; n<NMAX; ++n) { gavg+=g[n*blocksize+i]; } gavg/=NMAX;
    printf ("%.15le %.15le", i*h, gavg);
       for (n=0; n<NMAX; ++n) { printf (" %.15le",g[n*blocksize+i]); }
 
    
    for(r=0; r<((int)LMAX/2+1); r++){
    
        g2=0;
    
       for (ix=0; ix<LMAX; ++ix) {
        for (iy=0; iy<LMAX; ++iy) {
	    for (iz=0; iz<LMAX; ++iz) {
	   
           g2+=g[(ix * L2+iy *LMAX+iz)*blocksize+i]*
           (g[( ((ix+r)%LMAX) * L2+ iy*LMAX+ iz )*blocksize+i]+
           g[(ix * L2+  ((iy+r)%LMAX)*LMAX + iz)*blocksize+i]+
           g[( ix * L2+  iy * LMAX+((iz+r)%LMAX))*blocksize+i])/3;
	    }
	  }
	  
	}
    g2/=NMAX;
    
    printf (" %.15le %.15le", (float)r, g2);
    }
    printf ("\n");
    
  }
  fflush(stdout);    
}

/* output calculated values
   first two columns will be t and <g_n(t)>, after that all the correlation follows */

int main (int argc, const char **argv) {
  uint d, maxinit;
  char var [80]; double val;
  fp sigmaavg, minsig, maxsig, avg, sdev, vari;
  uint minsign, maxsign, n;
  unsigned long int rngseed;
  fp b;
  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);


  /* default values */
  h = 1e-5;
  blocksize = 256;
  blocks = 40;
  maxiter = 1000;
  maxinit = 80;
  accuracy = 1e-9;
 

  filter = 10; // output every 10th value only
//  filter =1; // you do the filtering at the level of the output function 
  lambda = 0.75;
  t0 = 1;
  sigmaavg = 0; // the average distance parameter
  sdev = 0; // this is going to be the standard deviation aroun sigmaavg
  alpha = 1;    // a scale for the gradient-dx

  rngseed = 0;  // seed for random-number generator

  /* if present, scan user-supplied values */
  if (argc>1) {
    for (d=1; d<argc; d++) {
      if (sscanf(argv[d], "%40[a-z0-9] = %lf", (char*)(&var), &val)==2) {
        if (!strcmp(var,"h")) h = val;
        else if (!strcmp(var,"blocksize")) blocksize = (uint)val;
        else if (!strcmp(var,"blocks")) blocks = (uint)val;
        else if (!strcmp(var,"maxiter")) maxiter = (uint)val;
        else if (!strcmp(var,"maxinit")) maxinit = (uint)val;
        else if (!strcmp(var,"accuracy")) accuracy = val;
        else if (!strcmp(var,"lambda")) lambda = val;
        else if (!strcmp(var,"sigma")) sigmaavg = val;
        else if (!strcmp(var,"sdev")) sdev = val;
        else if (!strcmp(var,"t0")) t0 = val;
        else if (!strcmp(var,"seed")) rngseed = val;
        else if (!strcmp(var,"alpha")) alpha = val;
        else if (!strcmp(var,"filter")) filter = (uint)val;
        else printf("ignored unknown variable %s\n", var);
      }
    }
  }

  // determine MCT exponents (b is just for information)
  a = regula_falsi (0.2, 0.3, exponent_func);
  b = -regula_falsi (-0.5, -0.1, exponent_func);
    
  A1= 1/(2 *(a * M_PI / sin(a*M_PI)-lambda));

  // initialize the array of distance parameters (Gaussian distribution)
  gsl_rng_set (r, rngseed);
  avg = 0;
  for (n=0; n<NMAX; ++n) {
    sigma[n] = gsl_ran_gaussian_ziggurat (r, sdev);
    avg += sigma[n];
  }
  avg /= NMAX;
 
  /*
  //below you have commented the part  sigma[n] += sigmaavg-avg;  that imposed that the sum of the random field should be equal exactly to sigma, because this causes troubles whenever the correlation is of the size of the system.
  */
  
  // since we are using a finite number of realizations,
  // we should ensure that the actually realized average is the one we want
  
  for (n=0; n<NMAX; ++n) {
    sigma[n] += sigmaavg; // - avg;
  }
  
  
  // re-check the average, collect information on biggest/smallest realization
  avg = 0;
  vari = 0;
  minsig = 1000; maxsig = -1000;
  for (n=0; n<NMAX; ++n) {
    avg += sigma[n];
    if (sigma[n]>maxsig) { maxsig=sigma[n]; maxsign=n; }
    if (sigma[n]<minsig) { minsig=sigma[n]; minsign=n; }
  }
  avg /= NMAX;
  for (n=0; n<NMAX; ++n) {
    vari += (sigma[n]-avg)*(sigma[n]-avg);
  }
  vari /= (NMAX-1);
  vari = sqrt(vari);

  halfblocksize = blocksize/2;

  g = (fp*)(malloc(NMAX*blocksize*sizeof(fp)));
  dG = (fp*)(malloc(NMAX*blocksize*sizeof(fp)));

  /*
  printf ("# beta correlator\n");
  #ifdef HG_VERSION
    #define xstr(s) str(s)
    #define str(s) #s
    printf ("# code revision %s\n", xstr(HG_VERSION));
  #endif
  printf ("# lambda = %lg\t sigma = %lg\t stddev(sigma) = %lg\n",
    lambda, sigmaavg, sdev);
  printf ("# t0 = %lg\n", t0);
  printf ("# a = %lg\t b = %lg\n", a,b);
  printf ("# (blocks,blocksize,h) = (%d,%d,%lg)\n", blocks,blocksize,h);
  printf ("# initial values, h0=%lg\n", h);
  #ifdef NABLA
    printf ("# N = %d realizations (with coupling, scale alpha=%lg):\n",
            NMAX, alpha);
  #else
    printf ("# N = %d realizations (without coupling):\n", NMAX);
  #endif
  printf ("# min/max sigma = %lg <%lg> %lg, stddev = %lg\n",
    minsig, avg, maxsig, vari);
  fprintf (stderr,
    "# min/max epsilon (n=%d, %d) : %lg <%lg> %lg (nominal %lg)\n",
    minsign, maxsign, minsig, avg, maxsig, sigmaavg);
  // dump individual rng values
  printf ("# dump of RNG values used: seed = %ld\n", rngseed);
  
 for (n=0; n<NMAX; ++n) {
    printf ("# n=%d\t%lg\n", n, sigma[n]);
  }
  printf ("#\n");

*/
  
//  printf ("# initial values, h0=%lg\n", h);
  initial_values (maxinit);
    outputcorr (0, maxinit); // you do  print the initial pure power-law values
//  printf ("# calculated values, h0=%lg\n", h);
  solve_block (maxinit, halfblocksize);
  outputcorr(maxinit, halfblocksize);  // you print only the last value see below
//  output (halfblocksize-1,halfblocksize);
  for (d=0; d<blocks; d++) {
//    printf ("# block %d: hd=%lg\n", d, h);
    solve_block (halfblocksize, blocksize);
    outputcorr(halfblocksize, blocksize);
 //   if(d<blocks-1) output(blocksize-1,blocksize);   // you print only the last value for each block, you must set filter to one.
    
 //   if(d<blocks) output(blocksize-1,blocksize);   // you print only the last value for each block, you must set filter to one.
 //   if(d==blocks-1) outputfull(blocksize-1,blocksize); 
    decimize ();
  }

  free (dG);
  free (g);
  gsl_rng_free (r);
  return (0);
}
