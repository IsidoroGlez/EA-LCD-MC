#include "../include/header.h"

int calculate_scalar_energy(int ibit, int irep,int iclon)
{
  int energia,x,y,z,site;
  int neigh_pz,neigh_px,neigh_py;
  char * aguja;
  char spin;

  energia=0;
  aguja=uu[ibit][irep][iclon];
  site=0;
  for(z=0;z<Lz;z++){
    neigh_pz=z_p[z];
    for(y=0;y<Ly;y++){
      neigh_py=y_p[y];
      for(x=0;x<Lx;x++){
	neigh_px=x_p[x];
	spin=aguja[site];

	energia+=spin^Jx[ibit][site]^aguja[site+neigh_px];
	energia+=spin^Jy[ibit][site]^aguja[site+neigh_py];
	energia+=spin^Jz[ibit][site]^aguja[site+neigh_pz];
	
	site++;
      }
    }
  }
  energia=3*V-2*energia;
  
  return energia;
}
