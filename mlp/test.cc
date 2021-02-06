//Contact: sylbarth@gmail.com, www.sylbarth.com
#include "mlp.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>


int main()
{
//FILE*   fp = NULL;
  int layers2[] = {15/*nb  entrees*/,5,5/*nb de neuronne dans la couche n+1*/,10/*nb sorties*/};
  MultiLayerPerceptron mlp2(4/*nb d'élement ds layer[]*/,layers2);
  mlp2.Run("C:/Users/alexandre/Documents/GitHub/mlp/AND2.dat",200000/*nb iteration*/);
  //mlp2.Test("C:/Users/alexandre/Documents/GitHub/mlp/AND2.dat");
 // mlp2.Simulate(in,out,tar,false);
  getchar();

  /*int layers2[] = {1,5,1};
  MultiLayerPerceptron mlp2(3,layers2);
  mlp2.Run("sin.dat",500);*/
  return 0;
}
