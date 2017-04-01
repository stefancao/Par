/**
 *  \file parallel-mergesort.cc
 *
 *  \brief Implement your parallel mergesort in this file.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "sort.hh"

/* function definitions */
void parallelMergeSort(keytype* A, int p, int r, keytype* B, int base);
void parallelMerge(keytype* T, int p1, int r1, int p2, int r2, keytype* A, int p3, int base);
void seqMergeSort(keytype* A, int p, int r, keytype* B);
void seqMerge(keytype* A, int p1, int r1, int p2, int r2, keytype* tmp, int np);
int binarySearch(int x, keytype* T, int p, int r);
int max(int a, int b);


// called in driver.cc
void parallelSort (int N, keytype* A)
{
	int p = 0;	// start index
	int r = N-1;	// end index

	// create another array
	keytype* B = newKeys (N);
	#pragma omp parallel
	{
		#pragma omp single
		{
			// using the base case as N/8, where we are assuming it is running on 8 threads
			parallelMergeSort(A, p, r, B, N/8);	
		}
	}
	free(B);	// free the array
}

void parallelMergeSort(keytype* A, int p, int r, keytype* B, int base) {
	
	// calculate the length
	int n = r - p + 1;

	// if the length is less than the base case, do sequential mergesort
	if (n <= base) {
		seqMergeSort(A, p, r, B);
		return;
	}
	
	/* Parallel MergeSort */
	// find the midpoint
	int q = p + (r - p)/2;

	// create a task to parallize the two mergesort, 
	// 		one takes first half, the other takes second half
	#pragma omp task
	{
		parallelMergeSort(A, p, q, B, base);
	}	
	parallelMergeSort(A, q+1, r, B, base);

	// wait for all the mergesort to be done
	#pragma omp taskwait	

	// call parallel merge
	parallelMerge(A, p, q, q+1, r, B, p, base);

	// copy the sorted elements in B into A
	memcpy(A + p, B + p, (r - p + 1) * sizeof(keytype));
}

void parallelMerge(keytype* T, int p1, int r1, int p2, int r2, keytype* A, int p3, int base) {
	
	// calculate the sizes of the two arrays
	int n1 = r1 - p1 + 1;
	int n2 = r2 - p2 + 1;

	// if the sizes of the two arrays added up together is less than the base case,
	// 		so sequential merge instead, to avoid overhead of context switching
	if((n1+n2) < base){
		seqMerge(T, p1, r1, p2, r2, A, p3);
		return;
	}

	// if the first array is smaller than the second array, we need to do some exchanges
	if (n1 < n2) {
		int tmp;
		
		// exchange p1 with p2
		tmp = p1;
		p1 = p2;
		p2 = tmp;

		// exchange r1 with r2
		tmp = r1;
		r1 = r2;
		r2 = tmp;

		// exchange n1 with n2
		tmp = n1;
		n1 = n2;
		n2 = tmp;
	}

	// make sure that the size of the first array is not 0
	if (n1 != 0) {

		// calculate the midpoint
		int q1 = (p1 + r1)/2;

		// find the index of the second half where everything on the left of it is 
		// 		less than the value of the midpoint of the first array
		int q2 = binarySearch(T[q1], T, p2, r2);

		// find the index that divides the subarrays
		int q3 = p3 + (q1 - p1) + (q2 - p2);
		A[q3] = T[q1];

		// spawn a new task to parallize the merge
		#pragma omp task
		{
			parallelMerge(T, p1 ,q1-1, p2, q2-1, A, p3, base);
		}
		parallelMerge(T, q1+1, r1, q2, r2, A, q3+1, base);

		// wait until all the merge are done
		#pragma omp taskwait
	}
}

int binarySearch(int x, keytype* T, int p, int r) {
	int low = p;
	int high = max(p, r+1);

	// find the index which all the values on left of it is less than the given value
	while (low < high) {
		int mid = (low+high)/2;
		if (x<=T[mid]) {
			high = mid;
		}
		else {
			low = mid + 1;
		}
	}
	return high;
}

int max(int a, int b) {
	if (a > b)
		return a;
	return b;
}

void seqMergeSort(keytype* A, int p, int r, keytype* B){
	int n = r - p + 1;

	// base case
	if(n<=1) return;

	// find the midpoint
	int q = p + (r - p)/2;
	seqMergeSort(A, p, q, B);
	seqMergeSort(A, q+1, r, B);
	seqMerge(A, p, q, q+1, r, B, p);

	// copy into A
	memcpy(A + p, B + p, (r - p + 1) * sizeof(keytype));
}

void seqMerge(keytype* A, int p1, int r1, int p2, int r2, keytype* tmp, int np){
	int i = p1;
	int j = p2;
	int ti = np;

	// do sorting by comparing the elements
	while(i<=r1 && j<=r2) {
		if (A[i] <= A[j]) {
			tmp[ti++] = A[i++];
		} else {
			tmp[ti++] = A[j++];
		}
	}
	while (i<=r1) { /* finish up lower half */
		tmp[ti++] = A[i++];
	}
	while (j<=r2) { /* finish up upper half */
		tmp[ti++] = A[j++];
	}
}

/* eof */
