

ANHANG: DAS PROGRAMM COVERINGS.C 71

Anhang: Das Programm coverings.c /* coverings.c -- (c)1997 Robert T. Firla (firla@math.tu-berlin.de)

berechnet fuer simpliziale Kegel eine Hilbert-Partition oder ein binaeres Hilbert-Cover bzw. stellt fest, dass keines existiert. */ /* Teil der Diplomarbeit

"Algorithmen fuer das Hilbert-Cover- und Hilbert-Partitions-Problem" an der TU Berlin bei Prof. Dr. Guenter M. Ziegler */

#include !stdio.h? #include !math.h? #include !malloc.h? #include "cplex.h"

/* eine Zeile der duennbesetzten Matrix */ typedef struct row.el -

int val; int column; struct row.el *next; struct row.el *prev;

"" row.el;

/* array von Listen, die die Zeilen der Matrix repraesentieren */ typedef struct ROWS - int nr.row.el; row.el *first; row.el *last;

"" ROWS;

typedef struct list.c.gen - int cone.nr; int *generators; struct list.c.gen *next;

"" list.c.gen;

double eps = 1e-12; /***********************************************************************/ /** EINIGE HILFSROUTINEN **/ /***********************************************************************/

int GCD(int a, int b) /* berechnet ggT(a,b) */ -

if (b == 0) return (a); else return (GCD(b,a%b)); ""

int gcd.array(int *A, int start, int nr) /* berechnet ggT(A.start,...,A.start+nr-1) */ -

int i, n = 2;

while (nr ?= 2)

-

for (i=0 ; i!=(int)ceil((double)(nr)/2)-1 ; i++)

if (n*i+1 ! nr)

A[n*i+1] = GCD(A[n*i+1],A[n*i+n/2+1]);

72 ANHANG: DAS PROGRAMM COVERINGS.C

nr = (int)ceil((double)(nr)/2); n = n*2; "" return (A[1]); ""

int* extended.euclid(int x, int y) /* Berechnet ggT(x,y) und ganze Zahlen p,q mit p*x + q*y = ggT(x,y).

Rueckgabe : int-array Gcd mit Gcd[0] = gcd, Gcd[1] = p, Gcd[2] = q */

-

int *Gcd,*Gcd.old; int a,b;

a = abs(x); b = abs(y); Gcd = (int*)calloc(3,sizeof(int)); if (b == 0)

-

Gcd[0] = a; Gcd[1] = 1; Gcd[2] = 0; return(Gcd); "" Gcd.old = extended.euclid(b, a%b); Gcd[0] = Gcd.old[0]; if (x ?= 0) Gcd[1] = Gcd.old[2]; else Gcd[1] = -Gcd.old[2]; if (y ?= 0) Gcd[2] = Gcd.old[1] - (int)floor((double)a/b)*Gcd.old[2]; else Gcd[2] = -(Gcd.old[1] - (int)floor((double)a/b)*Gcd.old[2]); return(Gcd); ""

int binomial(int n, int k) -

int i; double bin = 1;

for (i=1 ; i!=k ; i++)

bin = bin * (n-i+1)/i; return((int)bin); ""

int sign(int I) -

if (I?0) return(1); else if (I==0) return(0);

else return(-1); ""

double dabs(double a) -

if (a?=0.0) return(a); else return(-a); ""

int determinant(int **matrix, int DIM) -

ANHANG: DAS PROGRAMM COVERINGS.C 73

double **A = NULL; double *hilf = NULL; int i,j,k,pivot; double t, det = 1.0, factor = 1.0;

A = (double**)calloc(DIM, sizeof(double *)); for (i=0 ; i!DIM ; i++)

-

A[i] = (double*)calloc(DIM, sizeof(double)); for (j=0 ; j!DIM ; j++) A[i][j] = (double)matrix[i+1][j]; "" for (i=0 ; i!DIM ; i++)

-

pivot = i; for(j=i+1 ; j!DIM ; j++)

if (dabs(A[pivot][i]) ! dabs(A[j][i]))

pivot = j; if (pivot != i)

-

hilf = A[i]; A[i] = A[pivot]; A[pivot] = hilf; factor = factor * (-1); "" if (A[i][i] != 0.0)

-

for (j=i+1 ; j!DIM ; j++)

-

t = -A[j][i]/A[i][i]; for (k=i ; k!DIM; k++) A[j][k] = A[j][k] + t*A[i][k]; "" "" "" for (i=0 ; i!DIM ; i++)

det = det * A[i][i]; det = det * factor; for (i=0 ; i!DIM ; i++)

if (A[i]) free((char *) A[i]); if (A) free((char *) A);

if (dabs(det - anint(det)) ! 1e-8)

return((int)anint(det)); else

-

fprintf(stdout," Sorry, can't compute the dorrect determinant."n"); fprintf(stdout," --det - anint(det)-- = %e"n",det-anint(det)); exit(1); "" ""

/***********************************************************************/ /** ROUTINEN ZUM EINLESEN UND TRANSFORMIEREN DES KEGELS **/ /***********************************************************************/

int get.the.cone(int **CONE, int **CONE.old, int **U, int DIM) /* Hier wird der Kegel eingelesen, in Hermite-Normalform gebracht und der

kuerzeste Vektor auf jedem Extremalstrahl ermittelt. Rueckgabewert: -1

74 ANHANG: DAS PROGRAMM COVERINGS.C

(falls dim(C)!n), 1 (wenn transformiert wurde) oder 0 (alles ok) */ -

int **cone = NULL; int *A = NULL; int *Gcd = NULL; int i, j, k, gcd; int status = 0; int u.ii, u.ij, u.ji, u.jj, tmp1, tmp2;

for (i=1 ; i!=DIM ; i++)

-

fprintf(stderr,"the %d th point please: ",i); for (j=1;j!=DIM;j++)

-

scanf("%d",&CONE[i][j]); CONE.old[i][j] = CONE[i][j]; "" "" cone = (int**)calloc(1+DIM, sizeof(int*)); for (i=1 ; i!=DIM ; i++)

-

cone[i] = (int*)calloc(DIM, sizeof(int)); for (j=0 ; j!DIM ; j++)

cone[i][j] = CONE[i][j+1]; "" if (determinant(cone, DIM) == 0)

-

fprintf(stdout,"This isn't a full dimensional cone."n"); return(-1); ""

for (i=1 ; i!=DIM ; i++)

U[i][i] = 1;

/* bringt die Matrix in untere Dreiecksform */ for (j=2 ; j!=DIM ; j++)

for (i=1 ; i!j ; i++)

-

if (CONE[i][j] != 0)

-

Gcd = extended.euclid(CONE[i][i], CONE[i][j]); u.ii = Gcd[1]; u.ij = -CONE[i][j]/Gcd[0]; u.ji = Gcd[2]; u.jj = CONE[i][i]/Gcd[0]; for (k=1 ; k!=DIM ; k++)

-

tmp1 = CONE[k][i]*u.ii + CONE[k][j]*u.ji; tmp2 = CONE[k][i]*u.ij + CONE[k][j]*u.jj; CONE[k][i] = tmp1; CONE[k][j] = tmp2; tmp1 = U[k][i]*u.ii + U[k][j]*u.ji; tmp2 = U[k][i]*u.ij + U[k][j]*u.jj; U[k][i] = tmp1; U[k][j] = tmp2; "" status = 1; ""

ANHANG: DAS PROGRAMM COVERINGS.C 75

"" /* mache alle Diagonalelemente positiv */ for (i=1 ; i!=DIM ; i++)

if (CONE[i][i] ! 0)

-

for (j=1 ; j!= DIM ; j++)

-

CONE[j][i] = -CONE[j][i]; U[j][i] = -U[j][i]; "" status = 1; ""

/* transformiere so, dass das Diagonalelement das groesste ist */ for (i=1 ; i!=DIM ; i++)

for (j=1 ; j!i ; j++)

-

if ((CONE[i][j] ?= CONE[i][i] ---- CONE[i][j] ! 0))

-

tmp1 = (int)floor((double)CONE[i][j]/CONE[i][i]); for (k=1; k!=DIM ;k++)

-

CONE[k][j] = CONE[k][j] - tmp1 * CONE[k][i]; U[k][j] = U[k][j] - tmp1 * U[k][i]; "" status = 1; "" ""

/* suche den jeweils kuerzesten Punkt auf jedem Extremalstrahl */ A = (int*)calloc(DIM+1,sizeof(int)); for (i=1; i!=DIM; i++)

-

for (j=1 ; j!=i ; j++)

A[j] = CONE[i][j]; gcd = gcd.array(A,1,i); /* ggt dieser Zeile */ if (gcd ? 1)

-

for (j=1 ; j!=i ; j++) CONE[i][j] = CONE[i][j]/gcd; status = 1; "" ""

if (A) free ((char *) A); for (i=0 ; i!=DIM ; i++)

if (cone[i]) free ((char *) cone[i]); if (cone) free ((char *) cone);

return(status); ""

/***********************************************************************/ /* Routinen zur Konstruktion der minimalen Hilbertbasis des Kegels */ /***********************************************************************/

void find.points(int **pt, int **CONE, int *P, int *I, double *L,

76 ANHANG: DAS PROGRAMM COVERINGS.C

int DIM, int act); int sum.check(int *i, int *j, int *k, int DIM);

int** find.hilbert.basis(int **CONE, int DIM) - int **h.basis = NULL; /* Hilbertbasis */ int **pt = NULL; /* alle ganzzahligen Punkte des Polytops P */ int *P = NULL; int *I = NULL; double *L = NULL; int nr.basis.el; /* Anzahl Punkte in der Hilbertbasis */ int nr.pt.el = 1; /* Anzahl Punkte in der Menge pt */ int i,j,k,counter,status, indicator;

for (i=1 ; i!=DIM ; i++)

nr.pt.el = nr.pt.el*CONE[i][i]; pt = (int**)calloc(nr.pt.el, sizeof(int*)); for (i=0 ; i!nr.pt.el ; i++)

pt[i] = (int*)calloc(2+DIM, sizeof(int)); P = (int*)calloc(1+DIM, sizeof(int)); I = (int*)calloc(1+DIM, sizeof(int)); L = (double*)calloc(1+DIM, sizeof(double));

/* konstruiere alle Punkte in pt */ find.points(pt, CONE, P, I, L, DIM, DIM);

/* finde alle Punkte, die Summe zweier anderer sind */ /* markiere mit 0 - wenn der Punkt in der Basis ist und 1 - sonst */ counter = 1; for (i=1;i!nr.pt.el;i++)

-

indicator = 0; /* wenn es 1 ist, ist der Punkt Summe zweier anderer */ j=1; while((j!nr.pt.el) && (indicator==0)

&& (pt[j][DIM]!=(int)(floor)((double)(pt[i][DIM]/2+eps)))) - k=(pt[i][DIM]-pt[j][DIM])*(nr.pt.el/CONE[DIM][DIM]); if (k!j) k=j; while ((k!=i-1)&&(pt[k][DIM]!=pt[i][DIM]-pt[j][DIM])&&(indicator==0))

-

if (sum.check(pt[i],pt[j],pt[k],DIM)==1)

-

pt[i][0]=1; indicator=1; counter++; "" else k++;

"" j++; "" ""

nr.basis.el = nr.pt.el - counter + DIM; /* konstruiere jetzt die Hilbertbasis */ h.basis = (int**)calloc(nr.basis.el+1,sizeof(int*)); h.basis[0] = (int*)calloc(3,sizeof(int)); h.basis[0][0] = nr.basis.el;

ANHANG: DAS PROGRAMM COVERINGS.C 77 h.basis[0][1] = DIM; for (i=1 ; i!=nr.basis.el ; i++)

h.basis[i] = (int*)calloc(DIM,sizeof(int)); for (i=1 ; i!=DIM ; i++)

for (j=0 ; j!=DIM-1 ; j++)

h.basis[i][j] = CONE[i][j+1]; i = DIM+1; status = 1; for (j=1 ; j!nr.pt.el ; j++)

if (pt[j][0]==0)

-

if (pt[j][DIM+1] == 0) status = 0; for (k=0;k!=DIM-1;k++) h.basis[i][k] = pt[j][k+1];

i++; "" h.basis[0][2] = status; /* ist status=1, dann hat die Basis Hoehe 1 */

fprintf(stdout,""nThis is the complete hilbert-basis of the cone: "n"); for (i=1 ; i!=h.basis[0][0] ; i++)

-

fprintf(stdout,"(%d) ",i); for (j=0 ; j!h.basis[0][1];j++)

fprintf(stdout," %d",h.basis[i][j]); fprintf(stdout,""n"); "" fprintf(stdout,""n points in the basis = %d"n",h.basis[0][0]); if (h.basis[0][2] == 1)

fprintf(stdout," All point have the same height."n");

if (P) free((char *) P); if (L) free((char *) L); if (I) free((char *) I); for (i=0 ; i!nr.pt.el ; i++)

if (pt[i]) free((char *) pt[i]); if (pt) free((char *) pt);

return(h.basis); ""

void find.points(int **pt, int **CONE, int *P, int *I, double *L,

int DIM, int act) /* Diese Prozedur konstruiert sukzessive alle ganzzahligen Punkte in P.

Es wird an der n-Stelle begonnen und sich zur ersten vorgearbeitet. */ -

double sum; int i;

for (I[act] = P[act] ; I[act] != P[act] + CONE[act][act]-1 ; I[act]++)

-

if (act!=1)

-

sum = 0; for (i=act+1 ; i!=DIM ; i++)

sum += L[i]*CONE[i][act]; L[act] = (double)(I[act]-sum)/(double)CONE[act][act]; sum = 0; for (i=act ; i!=DIM ; i++)

78 ANHANG: DAS PROGRAMM COVERINGS.C

sum += L[i]*CONE[i][act-1]; P[act-1] = (int)ceil(sum-eps); find.points(pt, CONE, P, I, L, DIM, act-1); "" else

-

sum = 0; for (i=act+1 ; i!=DIM ; i++)

sum += L[i]*CONE[i][act]; L[act]=(double)(I[act]-sum)/(double)CONE[act][act]; sum = 0; for (i=1 ; i!=DIM ; i++)

sum += L[i]; if (fabs(sum - 1.0) ! eps)

pt[I[0]][DIM+1] = 1; for (i=1 ; i!=DIM ; i++)

pt[I[0]][i] = I[i]; I[0]++; "" "" ""

int sum.check(int *i, int *j, int *k, int DIM) /* ueberpruefe, ob pt[i]=pt[j]+pt[k], gib 1 zurUcke falls ja, 0 sonst */ -

int check=1, n=1;

while ((check==1) && (n!=DIM))

-

if (i[n]!=j[n]+k[n]) check=0; n++; "" return(check); ""

/***********************************************************************/ /** Konstruiere den generischen Punkt P **/ /***********************************************************************/

int* find.generic.point(int **CONE, int DIM) /* hier wird ein generischer Punkt g=(g.0,...g.DIM-1) konstruiert */ -

int *g.point = NULL; int i, k, R.k1, R.k2, sum;

g.point = (int*)calloc(1+DIM , sizeof(int)); for (k=DIM ; k?=1; k--)

-

R.k1 = CONE[k][k]; R.k2 = 0; sum = 0; for (i=k+2 ; i!=DIM; i++)

-

if (CONE[i][k] - CONE[i][k+1]?0)

if (CONE[i][k+1] != 0)

R.k1 += CONE[i][k] - CONE[i][k+1]; else

R.k2 += CONE[i][k];

ANHANG: DAS PROGRAMM COVERINGS.C 79

"" if (R.k2 != 0)

-

R.k2 += CONE[k][k] - 1; for (i=k+1 ; i!=DIM-1 ; i++)

sum += g.point[i]; "" g.point[k-1] = 1 + g.point[k] * R.k1 + sum * R.k2; "" fprintf(stdout," Generic point:"); for (i=0;i!DIM;i++)

fprintf(stdout,"%d ",g.point[i]); fprintf(stdout,""n");

return(g.point); ""

/***********************************************************************/ /** ROUTINEN ZUR ERZEUGUNG DER INZIDENZMATRIX (SPARSE MATRIX) **/ /***********************************************************************/

int P.in.cone(int *g.point, int **c.matrix, int DIM, int det); int F.in.cone(int *index.cone, int i.out, int **bound, int DIM); void new.matrix.element(ROWS *matrix, int row, int column, int det,

int index); int facet.index(int *index.cone, int p, int DIM, int nr.facets); void next.cone.indices(int *index.cone, int DIM); int** check.bound(int **h.basis, int *index.cone, int nr.facets,

int nr.basis.el, int DIM);

ROWS* construct.cone.facet.matrix(ROWS *matrix, int **h.basis,

int *g.point, list.c.gen **c.list, int *nr.cones, int nr.facets, int nr.basis.el, int DIM) /* Dies Routine erzeugt die gerichtete Kegel-Facetten-Inzidenzmatrix und

eine Liste c.list mit den Indizes aller unimodularen Kegel, die gefunden werden. */ -

int **c.matrix = NULL; /* aktuell bearbeiteter simplizial Kegel */ int **bound = NULL; int *index.cone = NULL; /* Indizes des aktuell bearbeiteten Kegel */ list.c.gen *cc.list = NULL; list.c.gen *actuell = NULL; int det, i, counter.uni.cones;

index.cone = (int*)calloc(2+DIM, sizeof(int)); c.matrix = (int**)calloc(1+DIM, sizeof(int*));

for (i=1 ; i!=DIM ; i++)

index.cone[i]=i; index.cone[DIM+1] = nr.basis.el+1; index.cone[0] = nr.basis.el;

bound = check.bound(h.basis, index.cone, nr.facets, nr.basis.el, DIM); for (i=1;i!=DIM;i++)

index.cone[i]=i; counter.uni.cones=0;

80 ANHANG: DAS PROGRAMM COVERINGS.C

while(index.cone[1] != nr.basis.el - DIM + 1)

-

for (i=1 ; i!=DIM ; i++)

c.matrix[i] = (int*)h.basis[index.cone[i]]; det = determinant(c.matrix, DIM); if ((det == 1) ---- (det == -1))

-

if (P.in.cone(g.point, c.matrix, DIM, det) == 1)

new.matrix.element(matrix, 0, counter.uni.cones, 1, 1); for (i=DIM ; i?=1 ; i--)

if (F.in.cone(index.cone, i, bound, DIM) == 1) new.matrix.element(matrix, facet.index(index.cone, i, DIM,

nr.facets), counter.uni.cones, det, i);

/* abspeichern der Indizes des Kegels in cc.list */ if (counter.uni.cones==0)

-

cc.list = (list.c.gen*)calloc(1,sizeof(list.c.gen)); actuell = cc.list; "" else

-

actuell-?next = (list.c.gen*)calloc(1,sizeof(list.c.gen)); actuell = actuell-?next; "" actuell-?cone.nr = counter.uni.cones; actuell-?generators = (int*)calloc(DIM,sizeof(int)); for (i=0 ; i!DIM ; i++)

actuell-?generators[i] = index.cone[i+1]; actuell-?next = NULL; counter.uni.cones++; "" next.cone.indices(index.cone,DIM); "" /*end while*/

for (i=0 ; i!=nr.basis.el ; i++)

if (bound[i]) free((char *) bound[i]); if (bound) free((char *) bound); if (c.matrix) free((char *) c.matrix); if (index.cone) free((char *) index.cone);

*nr.cones = counter.uni.cones; *c.list = cc.list;

return(matrix); ""

int P.in.cone(int *g.point, int **c.matrix, int DIM, int det) /* Diese Routine testet, ob der generische Punkt innerhalb des aktuellen

Kegels liegt und gibt eine 1 zurueck, falls dies der Fall ist. */ -

int *help = NULL; int ind = 1,wrong = 0; int i=1,j,k, DET;

while((ind==1) && (i!=DIM))

-

help = c.matrix[i]; /* tausche den i-ten Punkt aus */

ANHANG: DAS PROGRAMM COVERINGS.C 81

c.matrix[i] = (int*)g.point; DET = determinant(c.matrix, DIM); if (sign(det) * sign(DET) == -1)

ind=0; if (DET == 0)

wrong = 1;

c.matrix[i] = help; i++; "" if ((wrong == 1) && (ind == 1))

-

fprintf(stderr,"SORRY, BUT THIS IS NOT A GENERIC POINT."n"); for (j=1 ; j!=DIM ; j++)

-

for (k=0 ; k!DIM ; k++)

fprintf(stderr,"%d ",c.matrix[j][k]); fprintf(stderr,""n"); "" "" return(ind); ""

int F.in.cone(int *index.cone, int i.out, int **bound, int DIM) /* Diese Routine prueft, ob die Facette, die sich durch entfernen von

i.out aus dem Indexarray ergibt, eine innere Facette ist. Dazu wird bound entsprechend durchsucht. Rueckgabe: 1, falls F im Inneren ist */ -

int pr=1; int i,j,k,actuell,ind,ind1,start;

for (i=1 ; i!=DIM ; i++)

if (i != i.out) pr = pr * bound[index.cone[i]][0]; if (pr==0)

return(1); else

-

if (i.out==1) start = 2; else start = 1; i = 1; ind = 1; while ((ind==1) && (i!=bound[index.cone[start]][0]))

-

actuell = bound[index.cone[start]][i]; j = start+1; ind = 0; while ((j!=DIM) && (ind==0))

- if (j != i.out)

- ind1 = 1; for (k=1 ; k!=bound[index.cone[j]][0] ; k++)

if (bound[index.cone[j]][k]==actuell)

ind1 = 0; ind = ind1; "" j++;

"" i++; ""

82 ANHANG: DAS PROGRAMM COVERINGS.C

"" return(ind); ""

void new.matrix.element(ROWS *matrix, int row, int column, int det,

int index) /* Diese Routine f"ugt ein neues Matrixelement ein, mit Zeilennr. row,

Spaltennr. column und Wert det*(-1)^(i-1) */ -

if (matrix[row].nr.row.el == 0)

-

matrix[row].first = (row.el*)calloc(1,sizeof(row.el)); matrix[row].nr.row.el++; matrix[row].first-?val = det*(int)pow(-1,(index-1)); matrix[row].first-?column = column; matrix[row].first-?next = NULL; matrix[row].first-?prev = NULL; matrix[row].last = matrix[row].first; "" else

-

matrix[row].last-?next = (row.el*)calloc(1,sizeof(row.el)); matrix[row].nr.row.el++; matrix[row].last-?next-?val = det*(int)pow(-1,(index-1)); matrix[row].last-?next-?column = column; matrix[row].last-?next-?next = NULL; matrix[row].last-?next-?prev = matrix[row].last; matrix[row].last = matrix[row].last-?next; "" ""

int facet.index(int *index.cone, int p, int DIM, int nr.facets) /* Die Funktion berechnet den Index, sprich Zeilennummer einer Facette */ -

int i,index,sum;

sum=0; for (i=1;i!p;i++)

sum += binomial(index.cone[0]-index.cone[i],DIM-i); for (i=p+1;i!=DIM;i++)

sum += binomial(index.cone[0]-index.cone[i],DIM-i+1); index = nr.facets-sum; return(index); ""

void next.cone.indices(int *index.cone, int DIM) /* Diese Prozedur ermittelt anhand von index.cone die Indizes des

lexikographisch naechsten Kegel, der zu bearbeiten ist. */ -

int i, ind = 1, act = DIM;

while ((act ?= 1) && (ind == 1))

-

if (index.cone[act] == index.cone[act+1] - 1)

act--; else ind = 0; "" if (act == 0) act = 1;

ANHANG: DAS PROGRAMM COVERINGS.C 83

index.cone[act]++; for (i=act+1 ; i!=DIM ; i++)

index.cone[i] = index.cone[act] + i - act; ""

int** check.bound(int **h.basis, int *index.cone, int nr.facets,

int nr.basis.el, int DIM) /* Dies Funktion prueft, welche Basispunkte auf welchen Facetten des

Kegels C liegen. Es liefert fuer jeden Punkt folgende Information:

bound[i][0] = Anzahl Facetten , auf den Punkt i liegt bound[i][j] = ist die Nummer der j-ten Facette auf der i liegt */ -

int **c.matrix = NULL; int **bound = NULL; int i,j,sum,det;

bound = (int**)calloc(1+nr.basis.el, sizeof(int*)); for (i=1 ; i!= nr.basis.el ; i++)

bound[i] = (int*)calloc(DIM, sizeof(int)); c.matrix = (int**)calloc(1+DIM, sizeof(int*));

/* Bearbeitung der ersten n-Punkte (entsprechen den Erzeugenden von C)*/ for (i=1 ; i!=DIM ; i++)

-

bound[i][0]=DIM-1; for (j=1;j!=DIM-1;j++)

if (j!=DIM-i) bound[i][j] = j; else bound[i][j] = j+1; "" while((index.cone[1] != 2) && (index.cone[DIM] != nr.basis.el)) -

if (index.cone[DIM-1] != DIM)

-

for (i=1 ; i!=DIM ; i++)

c.matrix[i] = (int*)h.basis[index.cone[i]]; det = determinant(c.matrix, DIM); if (det == 0)

-

bound[index.cone[DIM]][0]++; sum = 0; for (i=1 ; i!DIM ; i++)

sum += index.cone[i]; bound[index.cone[DIM]][bound[index.cone[DIM]][0]]=

sum-((DIM-1)*(DIM)/2)+1; "" next.cone.indices(index.cone,DIM); "" else

next.cone.indices(index.cone,DIM); "" if (c.matrix) free((char *) c.matrix);

return(bound); ""

/***********************************************************************/ /** KONSTRUKTION EINER HILBERT-PARTITIONIERUNG **/ /***********************************************************************/

84 ANHANG: DAS PROGRAMM COVERINGS.C void construct.cplex.lp.file(ROWS *matrix, int nr.rows, int nr.colums,

int *rhs);

int* search.for.hilbert.partition(ROWS *matrix, int nr.rows,

int nr.columns, int *m.rhs) -

CPXENVptr cpxenv; struct cpxlp *lp = NULL; char *probname = "ex.lp"; int status; int numcols, numrows, objsen; double *obj = NULL, *rhs = NULL; char *sense = NULL; int *matbeg = NULL, *matcnt = NULL, *matind = NULL; double *matval = NULL; double *lb = NULL, *ub = NULL; double *rngval = NULL; char *objname = NULL, *rhsname = NULL; char **colname = NULL, **rowname = NULL; char *colnamestore = NULL, *rownamestore = NULL; int colspace = 0, rowspace = 0, nzspace=0; unsigned colnamespace = 0, rownamespace = 0; char *ctype = NULL; double *x; int *solution = NULL; int solstat; int cur.numcols; int i,j; char c;

/* Aufbau eine CPLEX-Eingabefiles */ construct.cplex.lp.file(matrix, nr.rows, nr.columns, m.rhs);

cpxenv = CPXopenCPLEX (&status); if ( cpxenv == NULL ) - char errmsg[1024];

fprintf (stdout, " Could not open CPLEX environment."n"); CPXgeterrorstring (cpxenv, status, errmsg); fprintf (stdout, " %s", errmsg); goto TERMINATE; ""

status = CPXsetintparam (cpxenv, CPX.PARAM.SCRIND, 0); if ( status != 0 ) -

fprintf (stdout,

" Failure to turn on screen indicator, error %d."n", status); goto TERMINATE; ""

status = CPXlpmread(cpxenv, probname, &numcols, &numrows, &objsen, &obj,

&rhs, &sense, &matbeg, &matcnt, &matind, &matval, &lb, &ub, &objname, &rhsname, &colname, &colnamestore, &rowname, &rownamestore, &colspace, &rowspace, &nzspace, &colnamespace, &rownamespace, &ctype); if (status) goto TERMINATE;

lp = CPXloadlpwnames(cpxenv, probname, numcols, numrows, objsen, obj,

ANHANG: DAS PROGRAMM COVERINGS.C 85

rhs, sense, matbeg, matcnt, matind, matval, lb, ub, rngval, colname, colnamestore, rowname, rownamestore, colspace, rowspace, nzspace, colnamespace, rownamespace); if (lp==NULL) -

fprintf(stdout," Can't load the problem."n"); goto TERMINATE; ""

status = CPXloadctype (cpxenv, lp, ctype); if (status) -

fprintf(stdout," Can't load the ctype."n"); goto TERMINATE; ""

status = CPXmipoptimize (cpxenv, lp); if (status) -

fprintf(stdout," Can't optimize the problem."n"); goto TERMINATE; ""

solstat = CPXgetstat (cpxenv, lp); fprintf(stdout,"Solution status %d."n",solstat);

cur.numcols = CPXgetnumcols (cpxenv, lp); x = (double*)malloc(cur.numcols*sizeof(double)); if ( x == NULL ) -

fprintf (stdout, " No memory for solution values."n"); goto TERMINATE; ""

status = CPXgetmx (cpxenv, lp, x, 0, cur.numcols-1); if (status) -

fprintf(stdout," Can't obtain the solution."n"); goto TERMINATE; ""

if (solstat == 101) /* es existiert eine ganzzahlige Opt.-Loesung */

-

solution=(int*)calloc(cur.numcols,sizeof(int)); for (i=0; i!cur.numcols; i++)

-

sscanf(colname[i]," %c %d",&c,&j); if (fabs(x[i]-1.0) ! 1e-12)

solution[j]=1; "" ""

TERMINATE: if (lp != NULL)

-

status = CPXunloadprob(cpxenv, &lp); if ( status )

fprintf (stdout, " CPXunloadprob failed, error code %d."n", status); "" if (x) free ((char *) x); if (obj) free ((char *) obj); if (rhs) free ((char *) rhs); if (sense) free ((char *) sense);

86 ANHANG: DAS PROGRAMM COVERINGS.C

if (matbeg) free ((char *) matbeg); if (matcnt) free ((char *) matcnt); if (matind) free ((char *) matind); if (matval) free ((char *) matval); if (lb) free ((char *) lb); if (ub) free ((char *) ub); if (objname) free ((char *) objname); if (rhsname) free ((char *) rhsname); if (rngval) free ((char *) rngval); if (colname) free ((char *) colname); if (colnamestore) free ((char *) colnamestore); if (rowname) free ((char *) rowname); if (rownamestore) free ((char *) rownamestore); if (ctype) free ((char *) ctype);

if (cpxenv != NULL)

-

status = CPXcloseCPLEX (&cpxenv); if ( status )

-

char errmsg[1024]; fprintf (stdout, " Could not close CPLEX environment."n"); CPXgeterrorstring (cpxenv, status, errmsg); fprintf (stdout, " %s", errmsg); "" "" return (solution); ""

void construct.cplex.lp.file(ROWS *matrix, int nr.rows, int nr.colums,

int *rhs) /* Diese Routine erzeugt aus den Daten in "matrix" und "rhs" ein

Gleichungssystem in lp-Form als Eingabe fuer CPLEX */

-

FILE *fp; row.el *actuell; int i,c;

fp = fopen("ex.lp","w"); fprintf(fp, "Minimize"n"n"); fprintf(fp, "Subject To"n"); c = 0; for (i=0 ; i!nr.rows ; i++)

-

if (matrix[i].nr.row.el ? 0)

-

c++; actuell = matrix[i].first; fprintf(fp, " c%d:", c); while (actuell != NULL) -

if (actuell-?val == 1)

fprintf(fp, " + x%d", actuell-?column); else

fprintf(fp, " - x%d", actuell-?column); actuell = actuell-?next; ""

ANHANG: DAS PROGRAMM COVERINGS.C 87

fprintf(fp, " = %d"n", rhs[i]); "" "" fprintf(fp, "Integers"n"); for (i=0 ; i!nr.colums ; i++)

fprintf(fp, " x%d", i); fprintf(fp, ""nEnd"n"); fclose(fp); ""

/***********************************************************************/ /** ROUTINEN ZUR BERECHNUNG EINES HILBERT-COVERS **/ /***********************************************************************/

void interchange(ROWS *matrix, int *rhs, int i, int j); void add.rows(ROWS *matrix, int i, int j); void insert.new.matrix.el(ROWS *matrix, int i, int j, row.el *i.act,

row.el *j.act); int v.product(ROWS ROW, int *solution, int rhs, int act);

ROWS* gauss.elimination(ROWS *matrix, int *rhs, int nr.rows, int nr.columns) /* Diese Funktion fuehrt auf der Matrix "matrix' eine Gausselimination

durch, wobei beachtet werden muss, dass ueber dem Koerper GF(2) gerechnet wird. */ -

int i, j, pivot;

for (i=0 ; i!nr.columns ; i++)

- /* suche nach einem Pivot-Element */

if ((matrix[i].nr.row.el?0) && (matrix[i].first-?column == i))

pivot = 1; else

pivot = 0; j=i; while ((pivot == 0) && (j!nr.rows-1))

-

j++; if ((matrix[j].nr.row.el?0) && (matrix[j].first-?column == i))

pivot = 1; "" /* Vorwaerts-Elimination */

if (pivot == 1) /* other */

-

if (j != i)

interchange(matrix,rhs,i,j); for (j=i+1; j!nr.rows; j++)

if ((matrix[j].nr.row.el?0) && (matrix[j].first-?column == i))

-

add.rows(matrix,i,j); rhs[j]=rhs[j]^rhs[i]; "" "" ""

/* Teilweise Rueckwaerts-Elimination - wird nur durchgefuehrt, wenn

A[i][i] = 0 und A[i][j] != 0 fuer ein j?i ist, damit ergibt sich fuer die Matrix: A[i][i] = 0 -? A[i][j] = 0 fur alle j */

88 ANHANG: DAS PROGRAMM COVERINGS.C i=0;

while (i!nr.columns)

-

if ((matrix[i].nr.row.el?0) && (matrix[i].first-?column!=i))

-

j=matrix[i].first-?column; if ((matrix[j].nr.row.el?0) && (matrix[j].first-?column==j))

- add.rows(matrix,j,i);

rhs[i]=rhs[i]^rhs[j]; ""

else

interchange(matrix,rhs,j,i); "" else

i++; "" return(matrix); ""

void interchange(ROWS *matrix, int *rhs, int i, int j) /* Diese Prozedur vertauscht die i-te und j-te Zeile der Matrix */ -

row.el *help.first, *help.last; int help.rhs,help.nr.row.el;

help.first = matrix[i].first; help.last = matrix[i].last; help.nr.row.el = matrix[i].nr.row.el; matrix[i].first = matrix[j].first; matrix[i].last = matrix[j].last; matrix[i].nr.row.el = matrix[j].nr.row.el; matrix[j].first = help.first; matrix[j].last = help.last; matrix[j].nr.row.el = help.nr.row.el; help.rhs = rhs[i]; rhs[i] = rhs[j]; rhs[j] = help.rhs; ""

void insert.new.matrix.el(ROWS *matrix, int i, int j, row.el *i.act,

row.el *j.act) /* Diese Prozedur erzeugt in der j-ten Zeile ein neues Element an der

Position vor j.act mit den Eintraegen von i.act */ -

row.el *new.el;

matrix[j].nr.row.el++; new.el =(row.el*)calloc(1, sizeof(row.el)); new.el-?val = 1; new.el-?column = i.act-?column;

new.el-?next = j.act; new.el-?prev = j.act-?prev; if (j.act-?prev == NULL) /* falls j.act das erste Element ist */

matrix[j].first = new.el; else /* falls j.act innerhalb der Liste ist*/

ANHANG: DAS PROGRAMM COVERINGS.C 89

j.act-?prev-?next = new.el; j.act-?prev = new.el; ""

void add.rows(ROWS *matrix, int i, int j) /* Diese Routine addiert die i-te Zeile der Matrix zur j-ten Zeile.

Dies geschieht ueber dem Koerper GF(2). */ -

row.el *i.act , *j.act; row.el *new.el , *help; int i.column, j.column;

i.act = matrix[i].first; j.act = matrix[j].first;

while ((i.act != NULL) && (j.act != NULL))

-

i.column=i.act-?column; j.column=j.act-?column;

if (i.column == j.column) /* da beide Eintraege vorhanden sind, muss j.act geloescht werden */

-

if (j.act-?prev == NULL) /* falls j.act das erste Element ist */ -

matrix[j].first = j.act-?next;

if (matrix[j].nr.row.el == 1)

matrix[j].last = NULL; else

matrix[j].first-?prev = NULL; ""

else

-

if (j.act-?next == NULL) /* falls j.act letztes Element ist */

-

matrix[j].last=j.act-?prev; matrix[j].last-?next = NULL; "" else /* falls j.act innerhalb der Liste ist */

-

j.act-?prev-?next = j.act-?next; j.act-?next-?prev = j.act-?prev; "" "" /* entferne jetzt j.act */

matrix[j].nr.row.el--; help = j.act; j.act = j.act-?next; i.act = i.act-?next; if (help) free ((char *) help); ""

if (i.column ! j.column) /* da es keinen Eintrag in Liste j gibt, f"uge neues Element ein */

-

insert.new.matrix.el(matrix, i ,j , i.act, j.act); i.act = i.act-?next; ""

90 ANHANG: DAS PROGRAMM COVERINGS.C

if (i.column ? j.column) /* hier muss nichts weiter getan werden */

j.act = j.act-?next; ""

/* Eine der Listen ist fertig bearbeitet. F"uge den Rest der i-ten

Liste an die j-te an, falls es einen Rest gibt. */ if ((j.act == NULL) && (i.act != NULL))

while (i.act!=NULL)

-

new.el = (row.el*)calloc(1, sizeof(row.el)); new.el-?val = 1; new.el-?column = i.act-?column; new.el-?next = NULL; if (matrix[j].nr.row.el == 0)

-

matrix[j].first = new.el; matrix[j].last = matrix[j].first; new.el-?prev = NULL;

"" else

- matrix[j].last-?next = new.el;

new.el-?prev = matrix[j].last; matrix[j].last = matrix[j].last-?next; "" matrix[j].nr.row.el++; i.act = i.act-?next; "" ""

int construct.solution(int *solution, ROWS *matrix, int *rhs,

int nr.rows, int nr.columns) /* Diese Funktion versucht eine Loesung fuer das transformierte

Gleichungssystem zu finden. Gelingt dies, wird sie in "solution" zurueckgegeben und status = 1 gesetzt, sonst ist status = 0 */ -

int i, m.ii; int status = 1;

for (i=nr.columns ; i!nr.rows ; i++)

if (rhs[i] == 1)

return(0); i = nr.columns - 1; while((status == 1) && (i ?= 0))

-

if (matrix[i].nr.row.el?0)

m.ii = 1; else

m.ii = 0; if (m.ii == 0)

if (rhs[i] ==1) return(0); else solution[i] = 0; else

solution[i] = v.product(matrix[i], solution, rhs[i], i); i--; ""

ANHANG: DAS PROGRAMM COVERINGS.C 91

return(status); ""

int v.product(ROWS ROW, int *solution, int rhs, int act) /* Diese Routine berechnet die naechste act-te Komponente des

Loesungsvektors. */ -

int sum = 0; row.el *row;

if (ROW.nr.row.el ? 0)

-

row = ROW.first; if (row-?column == act)

row = row-?next; while (row != NULL)

-

sum = sum ^ solution[row-?column]; row = row-?next; "" "" return(sum ^ rhs); ""

/***********************************************************************/ /** AUSGABE EINER LOESUNG DES PARTITIONS- ODER COVERING-PROBLEMS **/ /***********************************************************************/ void solution.output(int *solution, list.c.gen *c.list, int nr.columns,

int DIM) /* Diese Funktion ermittelt mittels des Loesungsvektors und der Liste

c.list alle Kegel, die zu der berechneten Ueberdeckung gehoeren. */ -

int i, j; list.c.gen *actuell = NULL;

actuell = c.list; for (i=0 ; i!nr.columns ; i++)

- if (solution[i] == 1)

-

fprintf(stdout, " x%d : (",i); for (j=0 ; j!DIM ; j++)

fprintf(stdout, "%d ", actuell-?generators[j]); fprintf(stdout,") Value: 1"n"); "" actuell = actuell-?next; "" fprintf(stdout, "All other variables in the range 0 - %d are 0"n",

nr.columns-1); ""

/***********************************************************************/ /*** H A U P T M E N U **/ /***********************************************************************/

void main(int argc, char *argv[]) -

int **CONE = NULL; /* der zu bearbeitende, transformierte Kegel C */

92 ANHANG: DAS PROGRAMM COVERINGS.C

int **CONE.old = NULL; /* der ursprungliche Kegel */ int **U = NULL; /* die Tarnsformationsmatrix */ int **h.basis = NULL; /* die Hilbertbasis von C */ ROWS *matrix = NULL; /* die (duenn besetzte) Inzidenzmatrix */

/* Zeilen : 0 ... nr.rows-1 */ /* Spalte : 0 ... nr.columns-1 */ int *rhs = NULL; /* recht Seite des Gleichungssystem */ int *solution = NULL; /* Loesungsvektor des LGS */ int *g.point = NULL; /* der generische Punkt g */ list.c.gen

*c.list = NULL; /* Liste mit den Indizes der unimodularen Kegel*/ int DIM; /* Dimension n */ int nr.basis.el; /* Anzahl Hilbertbasispunkte */ int nr.cones; /* Anzahl unimodulare Kegel */ int nr.facets; /* Anzahl moeglicher (!) Facetten */ int nr.rows; /* Anzahl der Zeilen der Matrix */ int nr.columns; /* Anzahl der Spalten der Matrix */ row.el *hilf = NULL; list.c.gen *actuell = NULL; int i,j, status;

/* einlesen und Transformation des zu bearbeitenden Kegels */ fprintf(stdout,"Dimension of the cone: "); scanf("%d", &DIM); CONE = (int**)calloc(1+DIM,sizeof(int*)); CONE.old = (int**)calloc(1+DIM,sizeof(int*)); U = (int**)calloc(1+DIM,sizeof(int*)); for (i=1 ; i!=DIM ; i++)

-

CONE[i] = (int*)calloc(1+DIM,sizeof(int)); CONE.old[i] = (int*)calloc(1+DIM,sizeof(int)); U[i] = (int*)calloc(1+DIM,sizeof(int)); "" status = get.the.cone(CONE, CONE.old, U, DIM); if (status == -1) exit(1); if (status == 1)

-

fprintf(stdout," The transformed generators of the cone:"n"); for (i=1;i!=DIM;i++)

-

fprintf(stdout,"(%d) ",i); for (j=1;j!=DIM;j++)

fprintf(stdout," %d",CONE[i][j]); fprintf(stdout,""n"); "" ""

/* konstruiere die Hilbertbasis des Kegels */ h.basis = find.hilbert.basis(CONE, DIM); nr.basis.el = h.basis[0][0];

/* berechne einen generischen Punkt */ g.point = find.generic.point(CONE, DIM);

/* Erzeugung der Kegel-Facetten-Inzidenmatrix, als sparse-matrix */ nr.facets = binomial(nr.basis.el, DIM-1); nr.rows = nr.facets + 1; matrix = (ROWS*)calloc(nr.rows, sizeof(ROWS));

ANHANG: DAS PROGRAMM COVERINGS.C 93

matrix = construct.cone.facet.matrix(matrix, h.basis, g.point,

&c.list, &nr.cones, nr.facets, nr.basis.el, DIM); nr.columns = nr.cones; rhs = (int*)calloc(nr.rows, sizeof(int)); rhs[0] = 1; fprintf(stdout," Problemsize: %d rows %d columns"n",nr.rows,nr.columns);

solution = search.for.hilbert.partition(matrix, nr.rows, nr.columns,

rhs); if (solution != NULL) /* dann existiert eine Partitionierung */

-

fprintf(stdout, " Here is a solution of the partition problem:"n"); solution.output(solution, c.list, nr.columns, DIM); "" else

-

fprintf(stdout, " There is no partition."n"); if (nr.columns ? nr.rows)

-

matrix = (ROWS*)realloc(matrix, nr.columns*sizeof(ROWS)); rhs = (int*)realloc(rhs, nr.columns*sizeof(int)); for (i=nr.rows ; i!nr.columns ; i++)

-

matrix[i].nr.row.el = 0; matrix[i].first = NULL; matrix[i].last = NULL; rhs[i] = 0; "" nr.rows = nr.columns; ""

matrix = gauss.elimination(matrix, rhs, nr.rows, nr.columns); solution = (int*)calloc(nr.columns, sizeof(int)); status = construct.solution(solution, matrix, rhs, nr.rows, nr.columns);

if (status == 1)

-

fprintf(stdout,

" Here is a solution of the binary covering problem."n"); solution.output(solution, c.list, nr.columns, DIM); "" else

fprintf(stdout, "There is no binary covering"n"); ""

/* gib allen im Laufe des Programms alloziierten Speicherplatz frei. */

if (g.point) free((char *) g.point); g.point = NULL; if (rhs) free((char *) rhs); rhs = NULL; if (solution) free((char *) solution); solution = NULL; for (i=0; i!nr.rows; i++)

-

while (matrix[i].nr.row.el ? 0)

-

hilf = matrix[i].first; matrix[i].first = matrix[i].first-?next; if (matrix[i].first) matrix[i].first-?prev = NULL;

94 ANHANG: DAS PROGRAMM COVERINGS.C

if (hilf)

-

hilf-?prev = NULL; hilf-?next = NULL; free((char *) hilf); hilf = NULL; "" matrix[i].nr.row.el--; "" matrix[i].first = NULL; matrix[i].last = NULL; "" if (matrix) free((char *) matrix); matrix = NULL; while (c.list != NULL)

-

actuell = c.list; c.list = c.list-?next; actuell-?next = NULL; if (actuell-?generators)

-

free((char *) actuell-?generators); actuell-?generators = NULL; "" if (actuell) free((char *) actuell); actuell = NULL; "" for (i=0; i!=nr.basis.el; i++)

if (h.basis[i]) free((char * ) h.basis[i]); if (h.basis) free((char *) h.basis); ""