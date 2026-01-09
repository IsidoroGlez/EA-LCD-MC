/******************************************************************************/
//
//  Adapted from GPU-based Swendsen-Wang multi-cluster algorithm for the simulation 
//  of classical spin systems
//
//  Copyright (C) 2015  Yukihiro Komura, Yutaka Okabe
//  RIKEN, Advanced Institute for Computational Science
//  Department of Physics, Tokyo Metropolitan University
//
//  This program is subject to the Standard CPC licence, 
//        (http://cpc.cs.qub.ac.uk/licence/licence.html)
//
//  Prerequisite: the NVIDIA CUDA Toolkit 5.0 or newer
//
//  Related publication:
//    Y. Komura and Y. Okabe, Comput. Phys. Commun. 183, 1155-1161 (2012).
//    doi: 10.1016/j.cpc.2012.01.017
//    Y. Komura,              Comput. Phys. Commun. 194, 54-58 (2015).
//    doi:10.1016/j.cpc.2015.04.015
//
//  We use the CUDA implementation for the cluster-labeling based on 
//  the workby Komura.
//    Y. Komura,             
//      Comput. Phys. Commun. 194, 54-58 (2015).
//
#include "HoudayerXEA.h"

__global__ void device_function_analysis_YK(unsigned long, unsigned int**[]);
__global__ void device_ReduceLabels        (unsigned long, unsigned int**[], int**[]);
__device__ unsigned int root_find          (unsigned int*, unsigned long int);
__global__ void device_function_spin_select(unsigned long, unsigned long, char**[], double *);
static unsigned long int h_nx, h_nz;  // linear system size
DEVICONST unsigned long int nx, nz;

unsigned long int h_nla;
unsigned long totrandom;
DEVICONST unsigned long int nla;
curandGenerator_t gen;
double* d_random_data;
__managed__ char** d_overlap_newAll[MAXNSAMPLES];
__managed__ int** d_bondAll[MAXNSAMPLES];
__managed__ unsigned int** d_labelAll[MAXNSAMPLES];

char **h_overlap_new[MAXNSAMPLES];
int **h_bondAll[MAXNSAMPLES];
unsigned int** h_labelAll[MAXNSAMPLES];

static unsigned long Samplegrid;
static unsigned long long int gridSW, gridUpdate;
static unsigned int threadsSW;
static unsigned int tnclone, clonetbm;

__global__ void spin_flip_YK_custom(const long n, const long clonethreshold, char **ds_uu[],
                                    char **d_overlap_newAll[], unsigned int** d_labelAll[]) {

	const unsigned int sampleId = blockIdx.y;   
	char ** __restrict__ d_overlap=ds_uu[sampleId];
	char ** __restrict__ d_overlap_new=d_overlap_newAll[sampleId];	
	unsigned int ** __restrict__ d_label=d_labelAll[sampleId];
	
	const unsigned long tid = (unsigned long)blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int iclone = tid/n;
	unsigned int ltid = tid - (iclone*n);
	int nover=1;
   	unsigned int offset=(iclone*nover); /* for the time being nover is always equal to 1 */
	if(offset>=clonethreshold) { if (ltid < n) { d_overlap[offset][ltid] = 0; } return; }  
	if (ltid < n) {
		if(d_overlap_new[offset][ ( d_label[offset][ltid] ) ] == 0x1) {
		   d_overlap[offset][ltid]^=0x1;  /* flip the value */
		}
	} else {
	  printf("Overlap flip custom Thread %lu out of %lu\n",tid,n);
	}
}

__global__ void init_YK_custom(const long n, char **ds_uu[], int** d_bondAll[], unsigned int** d_labelAll[]) {

	const unsigned int sampleId = blockIdx.y;   
	char ** __restrict__ d_overlap=ds_uu[sampleId];
	int ** __restrict__  d_bond=d_bondAll[sampleId];
	unsigned int ** __restrict__ d_label=d_labelAll[sampleId];
	unsigned char overlap;

	const unsigned long tid = (unsigned long)blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int iclone = tid/n;
	unsigned int ltid = tid - (iclone*n);
	int nover=1;
   	unsigned int offset=(iclone*nover); /* for the time being nover is always equal to 1 */	

	unsigned long la, index_min=ltid;
	int bond=0;
    	overlap = d_overlap[offset][ltid];
	if (ltid < n) {
	    for(int i=0; i<3; i++){
 	        if(i==0)la = (ltid-1+nx)%nx+((unsigned long)(ltid/nx))*nx;	
		if(i==1)la = (ltid-nx+(nx*nx))%(nx*nx) + (unsigned long)(ltid/(nx*nx))*nx*nx;
		if(i==2) {
#ifdef OBC
		  if(ltid<(nx*nx)) {
		    continue;
		  } else {
		    la = (ltid-(nx*nx)+nla)%nla;
		  }
#else /* PBC */
		  la = (ltid-(nx*nx)+nla)%nla;
#endif
		}
     		if( overlap == d_overlap[offset][la] ){
       	            bond |=  0x01<<i;
         	    index_min = min(index_min, la);
       	     	}
    	    }
            // Transfer "label" and "bond" to global memory
            d_bond[offset][ltid]  = bond;
    	    d_label[offset][ltid] = index_min;
	} else {
	  printf("init custom 1 Thread %llu out of %llu\n",tid,n);
	}
	
}

void ClusterInit(int nclone, double betathreshold, int nsample, unsigned int threads, unsigned long long seed) {

/*------------ Variables for GPU command ------------------------------*/
    h_nx=L;
    h_nz=Lz;
    h_nla = ((unsigned long)h_nx)*((unsigned long)h_nx)*((unsigned long)h_nz);
 
    MY_CUDA_CHECK( cudaMemcpyToSymbol(nx,&h_nx,sizeof(unsigned int),0,cudaMemcpyHostToDevice) );
    MY_CUDA_CHECK( cudaMemcpyToSymbol(nz,&h_nz,sizeof(unsigned int),0,cudaMemcpyHostToDevice) );    
    MY_CUDA_CHECK( cudaMemcpyToSymbol(nla,&h_nla,sizeof(unsigned long),0,cudaMemcpyHostToDevice) );
    
    threadsSW=threads;
    Samplegrid=nsample;
    clonetbm=lround(nclone*betathreshold);
    tnclone=nclone;
    if(clonetbm<1) {
    	fprintf(stderr,"Invalid beta threshold: %d %f\n",nclone,betathreshold);
	exit(1);
    }
    gridSW    = clonetbm*((h_nla+threadsSW-1)/((unsigned long int)threadsSW));
    if(gridSW>=((1LL<<31)-1)) {
       fprintf(stderr,"Invalid number of blocks: max is %d\n",(1LL<<31)-1);
       exit(1);
    }
    gridUpdate    = tnclone*((h_nla+threadsSW-1)/((unsigned long int)threadsSW));
    if(gridUpdate>=((1LL<<31)-1)) {
       fprintf(stderr,"Invalid number of blocks: max is %d\n",(1LL<<31)-1);
       exit(1);
    }

/*------------ Size of array ------------------------------------------*/

    unsigned long long int mem_rand              = sizeof(double)*(h_nla*nclone*nsample);
    totrandom=h_nla*nclone*nsample;

    for(int ibit=0;ibit<nsample;ibit++){
        h_overlap_new[ibit] = (char **)mmcuda((void ***)&d_overlap_newAll[ibit],nclone,L*L*Lz,sizeof(char),1);
        h_bondAll[ibit] = (int **)mmcuda((void ***)&d_bondAll[ibit],nclone,L*L*Lz,sizeof(int),1);
        h_labelAll[ibit] = (unsigned **)mmcuda((void ***)&d_labelAll[ibit],nclone,L*L*Lz,sizeof(unsigned),1);    		
    }


    MY_CUDA_CHECK( cudaMalloc((void**) &d_random_data, mem_rand) );
/*------------ Set of initial random numbers --------------------------*/

//
//  You can choose several random number generations: 
//  CURAND_RNG_PSEUDO_XORWOW(=CURAND_RNG_PSEUDO_DEFAULT), 
//  CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
//  CURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_MT19937
//
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    cudaDeviceSynchronize();
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)seed);
    cudaDeviceSynchronize();
    printf("###ClusterInit done: h_nla=%u, gridSW=%u\n",h_nla,gridSW);    
}    

void ClusterStep(char **d_overlapAll[]) {
//    Random number generation
#if defined(TEST_HOUDAYER_CONF)
     static int cnt=0;
     if((cnt&0x1)==0) {
#endif
      curandGenerateUniformDouble(gen, d_random_data, totrandom);
#if defined(TEST_HOUDAYER_CONF)
     }
     cnt++;
#endif
      MY_CUDA_CHECK(cudaDeviceSynchronize());
      dim3 ClusterGrid(gridSW,Samplegrid,1);
      dim3 UpdateGrid(gridUpdate,Samplegrid,1);      

      // Bond connection
      init_YK_custom<<<ClusterGrid, threadsSW>>>(h_nla, d_overlapAll, d_bondAll, d_labelAll);
      MY_CUDA_CHECK(cudaDeviceSynchronize());
      // Cluster formation
      device_function_analysis_YK<<<ClusterGrid,threadsSW>>>(h_nla, d_labelAll);
      device_ReduceLabels<<<ClusterGrid,threadsSW>>>(h_nla, d_labelAll, d_bondAll);
      device_function_analysis_YK<<<ClusterGrid,threadsSW>>>(h_nla, d_labelAll);
      
      device_function_spin_select<<<ClusterGrid,threadsSW>>>(h_nla, NBETAS, d_overlap_newAll, d_random_data);

      spin_flip_YK_custom<<<UpdateGrid, threadsSW>>>(h_nla, clonetbm, d_overlapAll, d_overlap_newAll, d_labelAll);
      MY_CUDA_CHECK(cudaDeviceSynchronize());
}
/*------------- Cleaning of memory ------------------------------------*/

void SWclose() { 
    curandDestroyGenerator(gen);
#if 0
    cudaFree(d_overlap_new);
    cudaFree(d_bond);
    cudaFree(d_label);
#endif    
    cudaFree(d_random_data);
}


/*****************************************************************************
        Device functions (GPU)
*****************************************************************************/


//****************************************************************************

//****************************************************************************
__global__ void device_function_analysis_YK(unsigned long n, unsigned int** d_labelAll[])
/*
     Equivalence chain of "label": repeat the calculation up to the point 
           such that d_label[d_label[...[index]...]] = d_label[...[index]...];
        (Komura algorithm)
*/
{
    const unsigned int sampleId = blockIdx.y;   
    unsigned int ** __restrict__ d_label=d_labelAll[sampleId];
	
    const unsigned long tid = (unsigned long)blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int iclone = tid/n;
    unsigned int ltid = tid - (iclone*n);
    int nover=1;
    unsigned int offset=(iclone*nover); /* for the time being nover is always equal to 1 */

    d_label[offset][ltid] = root_find(d_label[offset], ltid);
}


//****************************************************************************
__global__ void device_ReduceLabels(unsigned long n, unsigned int** d_labelAll[], 
                                                    int** d_bondAll[])
/*
     Label reduction method. 
     Please see the pseudo-code of Algorithm 1 in 
     Comput. Phys. Commun. 194, 54-58 (2015).
*/
{
     const unsigned int sampleId = blockIdx.y;
     int ** __restrict__  d_bond=d_bondAll[sampleId];
     unsigned int ** __restrict__ d_label=d_labelAll[sampleId];

     const unsigned long tid = (unsigned long)blockIdx.x*blockDim.x + threadIdx.x;
     const unsigned int iclone = tid/n;
     unsigned int ltid = tid - (iclone*n);
     int nover=1;
     unsigned int offset=(iclone*nover); /* for the time being nover is always equal to 1 */	

    unsigned long index = ltid;
    unsigned long la, i;
    unsigned int label_1, label_2, label_3, flag;
    unsigned int bond;

    bond = d_bond[offset][index];

    __syncthreads();

/*------------ Comparison with "label" of left and y-top sites ------------*/
    
    for(i=0; i<2; i++){
     flag = 0;

     label_1 = root_find(d_label[offset], index);
     if( (bond>>i)&0x01 == 1 ){
      if(i==0)la = (index-1+nx)%nx+((int)(index/nx))*nx;
      if(i==1)la = (index-nx+nx*nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
      label_2 = root_find(d_label[offset], la);
      if(label_1!=label_2)flag = 1;
     }

     if(label_1 < label_2){ 
      label_3=label_1; 
      label_1=label_2; 
      label_2=label_3; 
     }
     while( flag == 1 ){
      label_3 = atomicMin(&d_label[offset][label_1], label_2);
      if(label_3==label_2){ flag = 0;        }
      else if(label_3>label_2){ label_1=label_3;                 }
      else if(label_3<label_2){ label_1=label_2; label_2=label_3;}
     }
    }

/*------------ Comparison with "label" of z-top boudanry site ---------*/

   if(index< (nx*nx)){
     flag = 0;

     label_1 = root_find(d_label[offset], index);
     if( (bond>>2)&0x01 == 1 ){
      la = (index-nx*nx+nla)%nla;
      label_2 = root_find(d_label[offset], la);
      if(label_1!=label_2)flag = 1;
     }

     if(label_1 < label_2){ 
      label_3=label_1; 
      label_1=label_2; 
      label_2=label_3; 
     }
     while( flag == 1 ){
      label_3 = atomicMin(&d_label[offset][label_1], label_2);
      if(label_3==label_2){ flag = 0;        }
      else if(label_3>label_2){ label_1=label_3;                 }
      else if(label_3<label_2){ label_1=label_2; label_2=label_3;}
     }
   }
}


//****************************************************************************
__device__ unsigned int  root_find(
  unsigned int* d_label, unsigned long index )
/* 
     Equivalence chain of "label": repeat the calculation up to the point 
           such that d_label[d_label[...[index]...]] = d_label[...[index]...];
*/
{

    unsigned long ref, t_label;

    t_label = index;
    ref = d_label[t_label];

    while( ref != t_label){
      t_label = ref;
      ref = d_label[t_label];
    }

  return ref;
}


//****************************************************************************
__global__ void device_function_spin_select(unsigned long n, unsigned long nclone, char** d_overlap_newAll[],
                double* d_random_data)
/*
     Set a new spin state for each "label"
*/
{
     const unsigned int sampleId = blockIdx.y;
     char ** __restrict__  d_overlap_new=d_overlap_newAll[sampleId];

     const unsigned long tid = (unsigned long)blockIdx.x*blockDim.x + threadIdx.x;
     const unsigned int iclone = tid/n;
     unsigned int ltid = tid - (iclone*n);
     int nover=1;
     unsigned int offset=(iclone*nover); /* for the time being nover is always equal to 1 */	

     d_overlap_new[offset][ltid] = (d_random_data[(sampleId*n*nclone)+tid]<0.5);    
}

