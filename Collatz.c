#include <stdio.h>
#include <string.h> /* home of memset: void * memset ( void * ptr, int value, size_t num ); */
#include <stdlib.h> /* home of strtoull */

/* Sample input:
 * 
 * gcc -O3 -o Collatz Collatz.c (highly recommended)
 * Computer$ /Collatz 300000000
 * Longest Collatz sequence for first 300000000 positive integers found at 268549803 with length 964.
 */

unsigned long long getnext( unsigned long long x );

unsigned long long getnext( unsigned long long x )
{
	if (x == 1) { return 1; }
	if (x % 2 == 0) { return x/2; }
	return (3 * x + 1);
}

int main(int argc, char *argv[])
{
	unsigned long long biggestYet;
	int argind;

	for ( argind = 0; argind < argc; argind++ )
	{
		const unsigned long long intrange = 1 + strtoull(argv[argind], NULL, 10);
		unsigned long long *lengths = malloc( (intrange) * sizeof(unsigned long long)); 
		memset(lengths, 0, (intrange) * sizeof(unsigned long long));
		unsigned long long longestlen[] = {0, 0};
		unsigned long long seqindex;
		unsigned long long origdex;

		for (origdex = 2; origdex <= intrange; origdex++) 
		{
			seqindex = origdex;
			while ( seqindex > 1 )
			{
				lengths[ origdex ] += 1;
				seqindex = getnext(seqindex);

				if ( seqindex <= intrange )
				{
					if ( lengths[ seqindex ] > 0)
					{
						lengths[ origdex ] =  lengths[ origdex ] + lengths[ seqindex ];
						if ( lengths[origdex] > longestlen[1] ) { longestlen[0] = origdex; longestlen[1] = lengths[origdex]; }
						seqindex = 1;
						continue;
					}
				}
			}
		}
		if(longestlen[0] > 0 ){ printf("Longest Collatz sequence for first %llu positive integers found at %llu with length %llu. \n", intrange-1, longestlen[0], longestlen[1]); }
	free(lengths);
	}
	return 0;
}
