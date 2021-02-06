//Contact: sylbarth@gmail.com, www.sylbarth.com

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "mlp.h"

void InitializeRandoms()
{
  //  srand( (unsigned)time( NULL ) );
  srand(4711);
}

int RandomEqualINT(int Low, int High)
{
  return rand() % (High-Low+1) + Low;
}

double RandomEqualREAL(double Low, double High)
{
  return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}

MultiLayerPerceptron::MultiLayerPerceptron(int nl, int npl[]) :
  nNumLayers(0),
  pLayers(0),
  dMSE(0.0),
  dMAE(0.0),
  dEta(0.25),
  dAlpha(0.9),
  dGain(1.0),
  dAvgTestError(0.075)
{
  int i,j;
  /* --- cr�ation des couches */
  nNumLayers = nl;
  pLayers    = new Layer[nl];

  /* --- init des couches */
  for ( i = 0; i < nl; i++ )
    {
      /* --- cr�ation des neurones */
      pLayers[i].nNumNeurons = npl[i];
      pLayers[i].pNeurons    = new Neuron[ npl[i] ];
      /* --- init des neurones */
      for( j = 0; j < npl[i]; j++ )
	{
	  pLayers[i].pNeurons[j].x  = 1.0;
	  pLayers[i].pNeurons[j].e  = 0.0;
	  if(i>0)
	    {
	      pLayers[i].pNeurons[j].w     = new double[ npl[i-1] ];
	      pLayers[i].pNeurons[j].dw    = new double[ npl[i-1] ];
	      pLayers[i].pNeurons[j].wsave = new double[ npl[i-1] ];
	    }
	  else
	    {
	      pLayers[i].pNeurons[j].w     = NULL;
	      pLayers[i].pNeurons[j].dw    = NULL;
	      pLayers[i].pNeurons[j].wsave = NULL;
	    }
	}
    }
}

MultiLayerPerceptron::~MultiLayerPerceptron()
{
  int i,j;
  for( i = 0; i < nNumLayers; i++ )
    {
      if ( pLayers[i].pNeurons )
	{
	  for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	    {
	      if ( pLayers[i].pNeurons[j].w )
		delete[] pLayers[i].pNeurons[j].w;
	      if ( pLayers[i].pNeurons[j].dw )
		delete[] pLayers[i].pNeurons[j].dw;
	      if ( pLayers[i].pNeurons[j].wsave )
		delete[] pLayers[i].pNeurons[j].wsave;
	    }
	}
      delete[] pLayers[i].pNeurons;
    }
  delete[] pLayers;
}

void MultiLayerPerceptron::RandomWeights()
{
  int i,j,k;
  for( i = 1; i < nNumLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	    {
	      pLayers[i].pNeurons[j].w [k]    = RandomEqualREAL(-0.5, 0.5);
	      pLayers[i].pNeurons[j].dw[k]    = 0.0;
	      pLayers[i].pNeurons[j].wsave[k] = 0.0;
	    }
	}
    }
}

void MultiLayerPerceptron::SetInputSignal(double* input)
{
  int i;
  for ( i = 0; i < pLayers[0].nNumNeurons; i++ )
    {
      pLayers[0].pNeurons[i].x = input[i];
    }
}

void MultiLayerPerceptron::GetOutputSignal(double* output)
{
  int i;
  for ( i = 0; i < pLayers[nNumLayers-1].nNumNeurons; i++ )
    {
      output[i] = pLayers[nNumLayers-1].pNeurons[i].x;
    }
}

void MultiLayerPerceptron::SaveWeights()
{
  int i,j,k;
  for( i = 1; i < nNumLayers; i++ )
    for( j = 0; j < pLayers[i].nNumNeurons; j++ )
      for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	pLayers[i].pNeurons[j].wsave[k] = pLayers[i].pNeurons[j].w[k];
}

void MultiLayerPerceptron::RestoreWeights()
{
  int i,j,k;
  for( i = 1; i < nNumLayers; i++ )
    for( j = 0; j < pLayers[i].nNumNeurons; j++ )
      for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	pLayers[i].pNeurons[j].w[k] = pLayers[i].pNeurons[j].wsave[k];
}

/***************************************************************************/
/* calculate and feedforward outputs from the first layer to the last      */
void MultiLayerPerceptron::PropagateSignal()
{
  int i,j,k;

  /* --- la boucle commence avec la seconde couche */
  for( i = 1; i < nNumLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  /* --- calcul de la somme pond�r�e en entr�e */
	  double sum = 0.0;
	  for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	    {
	      double out = pLayers[i-1].pNeurons[k].x;
	      double w   = pLayers[i  ].pNeurons[j].w[k];
	      sum += w * out;
	    }
	  /* --- application de la fonction d'activation (sigmoid) */
	  pLayers[i].pNeurons[j].x = 1.0 / (1.0 + exp(-dGain * sum));
	}
    }
}

void MultiLayerPerceptron::ComputeOutputError(double* target)
{
  int  i;
  dMSE = 0.0;
  dMAE = 0.0;
  for( i = 0; i < pLayers[nNumLayers-1].nNumNeurons; i++)
    {
      double x = pLayers[nNumLayers-1].pNeurons[i].x;
      double d = target[i] - x;
      pLayers[nNumLayers-1].pNeurons[i].e = dGain * x * (1.0 - x) * d;
      dMSE += (d * d);
      dMAE += fabs(d);
    }
  /* --- erreur quadratique moyenne */
  dMSE /= (double)pLayers[nNumLayers-1].nNumNeurons;
  /* --- erreur absolue moyenne */
  dMAE /= (double)pLayers[nNumLayers-1].nNumNeurons;
}

/***************************************************************************/
/* backpropagate error from the output layer through to the first layer    */

void MultiLayerPerceptron::BackPropagateError()
{
  int i,j,k;
  /* --- la boucle commence � l'avant derni�re couche */
  for( i = (nNumLayers-2); i >= 0; i-- )
    {
      /* --- couche inf�rieure */
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  double x = pLayers[i].pNeurons[j].x;
	  double E = 0.0;
	  /* --- couche sup�rieure */
	  for ( k = 0; k < pLayers[i+1].nNumNeurons; k++ )
	    {
	      E += pLayers[i+1].pNeurons[k].w[j] * pLayers[i+1].pNeurons[k].e;
	    }
	  pLayers[i].pNeurons[j].e = dGain * x * (1.0 - x) * E;
	}
    }
}

/***************************************************************************/
/* update weights for all of the neurons from the first to the last layer  */

void MultiLayerPerceptron::AdjustWeights()
{
  int i,j,k;
  /* --- la boucle commence avec la seconde couche */
  for( i = 1; i < nNumLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nNumNeurons; j++ )
	{
	  for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
	    {
	      double x  = pLayers[i-1].pNeurons[k].x;
	      double e  = pLayers[i  ].pNeurons[j].e;
	      double dw = pLayers[i  ].pNeurons[j].dw[k];
	      pLayers[i].pNeurons[j].w [k] += dEta * x * e + dAlpha * dw;
	      pLayers[i].pNeurons[j].dw[k]  = dEta * x * e;
	    }
	}
    }
}

void MultiLayerPerceptron::Simulate(double* input, double* output, double* target, bool training)
{

  if(!input)  return;
  if(!target) return;

  /* --- on fait passer le signal dans le r�seau */
  SetInputSignal(input);
  PropagateSignal();
  if(output) GetOutputSignal(output);

  //if(output && !training) printf("test: %.2f %.2f %.2f = %.2f\n", input[0],input[1],target[0],output[0]);
  if(output && !training){
    printf("%.0f%.0f%.0f = %.0f%.0f \n", input[0],input[1],input[2],target[0],target[1]);
    printf("%.0f%.0f%.0f = %.0f%.0f \n", input[3],input[4],input[5],target[2],target[3]);
    printf("%.0f%.0f%.0f = %.0f%.0f \n", input[6],input[7],input[8],target[4],target[5]);
    printf("%.0f%.0f%.0f = %.0f%.0f \n", input[9],input[10],input[11],target[6],target[7]);
    printf("%.0f%.0f%.0f = %.0f%.0f \n", input[12],input[13],input[14],target[8],target[9]);

    printf("je suis un 0 a %.2f%% 1 a %.2f%% 2 a %.2f%% 3 a %.2f%% 4 a %.2f%% \n", output[0]*100,output[1]*100,output[2]*100,output[3]*100,output[4]*100);
    printf("je suis un 5 a %.2f%% 6 a %.2f%% 7 a %.2f%% 8 a %.2f%% 9 a %.2f%% \n", output[5]*100,output[6]*100,output[7]*100,output[8]*100,output[9]*100);
  }

  /* --- calcul de l'erreur en sortie par rapport � la cible */
  /*     ce calcul sert de base pour la r�tropropagation     */
  ComputeOutputError(target);

  /* --- si c'est un apprentissage, on fait une r�tropropagation de l'erreur */
  if (training)
    {
      BackPropagateError();
      AdjustWeights();
    }
}

bool read_number(FILE* fp, double* number)
{
  char szWord[256];
  int i = 0;
  int b;

  *number = 0.0;

  szWord[0] = '\0';
  while ( ((b=fgetc(fp))!=EOF) && (i<255) )
    {
      if( (b=='.') ||
	  (b=='0') ||
	  (b=='1') ||
	  (b=='2') ||
	  (b=='3') ||
	  (b=='4') ||
	  (b=='5') ||
	  (b=='6') ||
	  (b=='7') ||
	  (b=='8') ||
	  (b=='9') )
	{
	  szWord[i++] = (char)b;
	}
      else
	if(i>0) break;
    }
  szWord[i] = '\0';

  if(i==0) return false;

  *number = atof(szWord);

  return true;
}

int MultiLayerPerceptron::Train(const char* fname)
{
  int count = 0;
  int nbi   = 0;
  int nbt   = 0;
  double* input  = NULL;
  double* output = NULL;
  double* target = NULL;
  FILE*   fp = NULL;

  fp = fopen(fname,"r");

  if(!fp){ printf("J'arrive pas à ouvrir le fichier %s",fname); return 0;}

  input  = new double[pLayers[0].nNumNeurons];
  output = new double[pLayers[nNumLayers-1].nNumNeurons];
  target = new double[pLayers[nNumLayers-1].nNumNeurons];
if(!input)  {printf("input   null"); return 0;}
if(!output) {printf("output  null"); return 0;}
if(!target) {printf("target  null"); return 0;}

  //printf("Train: input[%f],output[%f],target[%f]\n",input,output,target);
  while( !feof(fp) )
    {
      double dNumber;
      if( read_number(fp,&dNumber) )
	{
	  /* --- on le transforme en input/target */
	if( nbi < pLayers[0].nNumNeurons ){
        input[nbi++] = dNumber;
	    //printf("input[%i][%d]_",nbi,dNumber);
    }
	  else if( nbt < pLayers[nNumLayers-1].nNumNeurons ){
	    target[nbt++] = dNumber;
        //printf("target[%i][%d]_\n",nbt,dNumber);
	  }
	  /* --- on fait un apprentisage du r�seau  avec cette ligne*/
	  if( (nbi == pLayers[0].nNumNeurons) &&
	      (nbt == pLayers[nNumLayers-1].nNumNeurons) )
	    {

	      Simulate(input, output, target, true);
	      nbi = 0;
	      nbt = 0;
	      count++;
	    }
	}
      else
	{
	  break;
	}
    }

  if(fp) fclose(fp);

  if(input)  delete[] input;
  if(output) delete[] output;
  if(target) delete[] target;

  return count;
}

int MultiLayerPerceptron::Test(const char* fname)
{
  int count = 0;
  int nbi   = 0;
  int nbt   = 0;
  double* input  = NULL;
  double* output = NULL;
  double* target = NULL;
  FILE*   fp = NULL;

  fp = fopen(fname,"r");
  if(!fp){ printf("!!!!!!eerr\n");return 0;}

  input  = new double[pLayers[0].nNumNeurons];
  output = new double[pLayers[nNumLayers-1].nNumNeurons];
  target = new double[pLayers[nNumLayers-1].nNumNeurons];

  if(!input) return 0;
  if(!output) return 0;
  if(!target) return 0;

  dAvgTestError = 0.0;

  while( !feof(fp) )
    {
      double dNumber;
      if( read_number(fp,&dNumber) )
	{
	  /* --- on  le transforme en input/target */
	  if( nbi < pLayers[0].nNumNeurons )
	    input[nbi++] = dNumber;
	  else if( nbt < pLayers[nNumLayers-1].nNumNeurons )
	    target[nbt++] = dNumber;

	  /* --- on fait un apprentisage du réseau  avec cette ligne*/
	  if( (nbi == pLayers[0].nNumNeurons) &&
	      (nbt == pLayers[nNumLayers-1].nNumNeurons) )
	    {
	      Simulate(input, output, target, false);
	      dAvgTestError += dMAE;
	      nbi = 0;
	      nbt = 0;
	      count++;
	    }
	}
      else
	{
	  break;
	}
    }

  dAvgTestError /= count;

  if(fp) fclose(fp);

  if(input)  delete[] input;
  if(output) delete[] output;
  if(target) delete[] target;

  return count;
}

int MultiLayerPerceptron::Evaluate()
{
  int count = 0;
  return count;
}

void MultiLayerPerceptron::Run(const char* fname, const int& maxiter)
{
  int    countTrain = 0;
  int    countLines = 0;
  bool   Stop = false;
  bool   firstIter = true;
  double dMinTestError = 00;
 // double dAvgTestError = 0.075;
  InitializeRandoms();/* --- init du g�n�rateur de nombres al�atoires  */
  RandomWeights();  /* --- et g�n�ration des pond�rations al�atoires */
  /* --- on lance l'apprentissage avec tests */
  do {
    countLines += Train(fname);
    Test(fname);
    countTrain++;

    if(firstIter)
    {
        dMinTestError = dAvgTestError;
        firstIter = false;
        printf(" -> 1st iter\n");
    }
    printf( "Train nb[%i] dAvgTestError[%.2f] dMinTestError[%.2f]\n", countTrain, dAvgTestError,dMinTestError);
    if ( dAvgTestError < dMinTestError){
        printf(" -> saving weights\n");
        dMinTestError = dAvgTestError;
        SaveWeights();
//if(dMinTestError<0.02){Stop = true;}
    }
    else if (dAvgTestError > 1.2 * dMinTestError){
        printf(" -> stopping training and restoring weights\n");
        Stop = true;
        RestoreWeights();
    }
    else{
        printf(" -> ok\n");
    }
    //printf("\n");printf("TEST>>>: input=%f,output=%f,target=%f\n",input,output,target);

  } while ( (!Stop) && (countTrain<maxiter) );

  printf("byef\n");
}
