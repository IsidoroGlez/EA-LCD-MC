#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <sys/types.h>

#include "random_generator.h"

// DEFINITIONS

#define MAXSTR 1024
#define NR 8
#define MAXIT_TRW 512
#define MAXNBIN 4096
#define WARPSIZE 32
#define CUDATHREADS 512
#define MAXNSAMPLES 32

// CHECK DEFINES

#ifndef NUMBITSPREBUSQUEDAS
#error "NUMBITSPREBUSQUEDAS must be defined"
#endif

#define NPREBUSQUEDAS (1<<NUMBITSPREBUSQUEDAS)

#ifndef NBETAS
#error "NBETAS must be defined"
#endif

#ifndef L
#error "L must be defined"
#endif

#ifndef Lz
#error "M must be defined"
#endif

#if(((L>>3)<<3)!=L)
#error "L must be a multiple of 8"
#endif

#if(((Lz>>3)<<3)!=Lz)
#error "Lz must be a multiple of 8"
#endif

// PHYSICAL SYSTEM VARIABLES
#define DIM 3
#define DEGREE (DIM<<1)
#define Lx L
#define Ly L
#define V (Lx*Ly*Lz)
#define S (Ly*Lx)

#define NRNBETAS (NR*NBETAS)

//MSC SYSTEM VARIABLES
#define MSC_L (L>>2)
#define MSC_Lz (Lz>>2)
#define MSC_S (MSC_L*MSC_L)
#define MSC_V (MSC_S*MSC_Lz)
#define HALF_MSC_V (MSC_V>>1)
#define HALF_MSC_VNR (HALF_MSC_V*NR)
#define HALF_MSC_VNRNBETAS (HALF_MSC_VNR*NBETAS)
#define MSC_VDEGREE (MSC_V*DEGREE)

enum{MAX=65536};

// NEW VARIABLE TYPES

typedef unsigned long long int MYWORD;
#define BITSINMYWORD 64

#ifdef MAIN
typedef struct{uint4 vec[NR];} s_time;
typedef struct{uint32_t my_key[NR];} s_keys;
#else
// CUDA types in C
typedef struct{uint32_t x,y,z,w;} uint4;

typedef struct{uint4 vec[NR];} s_time;
typedef struct{uint32_t my_key[NR];} s_keys;
#endif


#define NDATINT 9
#define NDATRAND 4

typedef struct
{
  int nbin,
    itcut,
    itmax,
    houfr,
    mesfr,
    ptfr,
    nbetas,
    flag,
    l;
  randint seed_J,
    seed_u,
    seed_MC,
    seed_Cluster;
} s_data;

typedef struct{
  uint4 umbrales[128];
  uint4 prebusqueda[NPREBUSQUEDAS>>3];
} s_lut_heat_bath;

struct Vicini {
    MYWORD J[MAXNSAMPLES][DEGREE];
    int neig[DEGREE];
} ;

// VARIABLES
#ifdef MAIN
int list_samples[MAXNSAMPLES];

int x_p[Lx],x_m[Lx],y_p[Ly],y_m[Ly],z_p[Lz],z_m[Lz];

Vicini viciniB[HALF_MSC_V], viciniN[HALF_MSC_V];
unsigned int bianchi_index[HALF_MSC_V], neri_index[HALF_MSC_V];
unsigned char bianchi_rotate[HALF_MSC_V], neri_rotate[HALF_MSC_V];
int side[DIM], off[DIM+1]; 

int numThreadsPerBlock, numBlocks;

s_data data;
int write_seeds;
double betas[NBETAS];
int countJ0[MAXNSAMPLES];

s_lut_heat_bath h_LUT[NBETAS];

s_keys seed_keys;
s_time s_time_and_entropy;
randint seeds_J[MAXNSAMPLES], seeds_u[MAXNSAMPLES], seeds_MC[NR];
s_aleatorio_HQ_64bits random_u, random_J;
s_aleatorio_HQ_64bits random_PRC[NR], random_PT[MAXNSAMPLES][NR];
s_xoshiro256pp random_xoshiro256pp[NR];

char ***uu[MAXNSAMPLES], *Jx[MAXNSAMPLES], *Jy[MAXNSAMPLES], *Jz[MAXNSAMPLES];
MYWORD *h_MSC_u_even[MAXNSAMPLES], *h_MSC_u_odd[MAXNSAMPLES];
MYWORD **h_J[MAXNSAMPLES], **h_spin[MAXNSAMPLES], **h_overlap[MAXNSAMPLES];
char **h_uu[MAXNSAMPLES];
int **h_neig[MAXNSAMPLES];
int *h_Ener;

double aceptancePT[MAXNSAMPLES][NR][NBETAS-1], attemptsPT[MAXNSAMPLES][NR][NBETAS-1], aceptancePTraw[MAXNSAMPLES][NR][NBETAS-1];
uint8_t which_clon_this_beta[MAXNSAMPLES][NR][NBETAS];
uint8_t which_beta_this_clon[MAXNSAMPLES][NR][NBETAS];

double average_Energy[MAXNSAMPLES][NRNBETAS];

int frecTRW, maxit_TRW;
unsigned char history_betas[MAXNSAMPLES][NR][MAXIT_TRW][NBETAS];

int sum[MAX];

// DEVICE VARIABLES
#define DEVICONST __device__ __constant__
#define DEVIBASE __device__

DEVICONST int d_deltaE, d_deltaN, d_deltaU, d_deltaO, d_deltaS, d_deltaD;
unsigned int *d_bianchi_index, *d_neri_index;
unsigned char *d_bianchi_rotate, *d_neri_rotate;

__managed__ MYWORD **d_J[MAXNSAMPLES];
__managed__ int **d_neig[MAXNSAMPLES];
__managed__ MYWORD  **d_spin[MAXNSAMPLES];
__managed__ MYWORD  **d_overlap[MAXNSAMPLES];
__managed__ char  **ds_uu[MAXNSAMPLES];

__managed__ unsigned int *d_whichclone[MAXNSAMPLES];
__managed__ unsigned int *d_whichclonethisbeta[MAXNSAMPLES];

s_lut_heat_bath *dev_lut_heat_bath;
uint64_t *rand_wht_h, *rand_blk_h;

int *d_Ener;
int *d_sum;

#else
extern int list_samples[];

extern int x_p[],x_m[],y_p[],y_m[],z_p[],z_m[];

extern Vicini viciniB[HALF_MSC_V], viciniN[HALF_MSC_V];
extern unsigned int bianchi_index[], neri_index[];
extern unsigned char bianchi_rotate[], neri_rotate[];
extern int side[], off[]; 

extern int numThreadsPerBlock, numBlocks;

extern s_data data;
extern int write_seeds;
extern double betas[];
extern int countJ0[];

extern s_lut_heat_bath h_LUT[];

extern s_keys seed_keys;
extern randint seeds_J[], seeds_u[], seeds_MC[];
extern s_aleatorio_HQ_64bits random_u, random_J;
extern s_aleatorio_HQ_64bits random_PRC[], random_PT[][NR];
extern s_xoshiro256pp random_xoshiro256pp[NR];

extern char ***uu[MAXNSAMPLES], *Jx[MAXNSAMPLES], *Jy[MAXNSAMPLES], *Jz[MAXNSAMPLES];
extern MYWORD *h_MSC_u_even[MAXNSAMPLES], *h_MSC_u_odd[MAXNSAMPLES];
extern int *h_Ener;

extern double aceptancePT[][NR][NBETAS-1], attemptsPT[][NR][NBETAS-1], aceptancePTraw[][NR][NBETAS-1];
extern uint8_t which_clon_this_beta[][NR][NBETAS];
extern uint8_t which_beta_this_clon[][NR][NBETAS];

extern double average_Energy[][NRNBETAS];

extern int frecTRW, maxit_TRW;
extern unsigned char history_betas[][NR][MAXIT_TRW][NBETAS];

extern int sum[];
#endif

// FUNCTIONS PROTOTYPES

// IO:
void write_conf(int,int);
void read_conf(int);
void Janus_packing_for_write(int, int);
void Janus_unpacking_for_read(int, int);
void write_measures(int, int);
void write_history(int, int);
void read_input(const char *);
void check_data(s_data);
void print_data(FILE *, s_data *);
void read_betas(const char *);
void read_lut(const char *, const char *);
void read_list_samples(const char *, int);
void create_sample_path(int, int);
void check_and_prepare_simulation_backup(int, int);
int get_seeds(int);
void write_seeds_in_file(int);
void backup(int);

void stop(int *);
void ignore_sigterm(void);
void handle_sigterm(void);
void stop_run(int);
void create_error(const char *format, ...);
void create_running(void);
void renew_running(void);
void delete_running(void);
void writelog(FILE *, time_t);
void print_help(const char *);
void print_and_exit(const char *format, ...);

// TIMING
void measure_time(int);
int check_time(unsigned long long);

// INIT:
void Init(int);
void generate_seed_vectors(void);
void Init_Random(int);
void Init_neighbours(void);
void Set_MSC_neigh(unsigned int *, int,int, int);
int punto(int *, int *);
void coordinate(int, int *,int *);
void sp(int *, int * , int *, int);
void Init_MSC_neighbours(void);
void Init_MSC_index_and_rotate(void);
void Init_Binario(void);

void Init_u(int);
void packing_u(int);
void unpacking_u(int);
void check_packing(int);

void Init_J(int);
void packing_J(int);

void calculate_blocks(void);

void Init_PT(int);
void init_tempering(int);
void init_aceptances(int);

// UPD:
void Parallel_Tempering(s_aleatorio_HQ_64bits *, uint8_t *,uint8_t *,
			int *, double *, double *, double *);

// MED:
int calculate_scalar_energy(int, int, int);

// MAIN:
int Monte_Carlo(uint32_t, int, int);
void Houdayer(int);
void Measure_Energy(int);

// HOST MEMORY
void AllocHost(int);
void FreeHost(int);

// DEVICE MEMORY
void CopyConstOnDevice(void);
void CopyDataOnDevice(int);
void send_PT_state_to_GPU(int);
void FreeDevice(int);

// KERNELS
#ifdef MAIN
__global__ void d_gen_rndbits_cuda(int, int, int, s_time, s_keys, s_lut_heat_bath *, uint64_t *);

__global__ void d_computeEne(MYWORD *, MYWORD *, int *, int *, int, MYWORD **, int **, int,
			     int, int, unsigned int *, unsigned int *);

__global__ void d_oneMCstepBN_multisample(MYWORD **spinAll[],
					  const int dir,
					  MYWORD **JAll[],
					  int **neigAll[],
					  int n,
					  int nrep,
					  int nclone,
					  unsigned int **whichcloneAll,
					  const unsigned char *rotate,
					  uint64_t *rand_h);

__global__ void d_computeoverlap(MYWORD **spinAll[],
			  MYWORD **overlapAll[],
			  int n,
			  int nrep,
			  int nclone,
			  int r1,
			  int r2,
			  unsigned int **whichcloneAll);

__global__ void d_useoverlap(MYWORD **spinAll[],
			  MYWORD **overlapAll[],
			  int n,
			  int nrep,
			  int nclone,
			  int r1,
			  int r2,
			  unsigned int **whichcloneAll);

__global__ void d_pack(MYWORD **overlapAll[],
		       char **overlapu[],
		       int n,
		       int nrep,
		       int nclone);

__global__ void d_unpack(MYWORD **overlapAll[],
		       char **overlapu[],
		       int n,
		       int nrep,
		       int nclone);


#endif

void **mmcuda(void ***, int , int , int , int);
