// Author: Alex Walczak, 2015.
import java.text.DecimalFormat;

// Euler Project #14 Solution.
// Fast calculation of the integer with the longest Collatz sequence of a set of integers from 1 to max,
// and the length of this sequence.
// Crashes when maxint exceeds 178,913,277. Finds longest sequence at 169,941,673 with length 953.

/** SAMPLE INPUT AND OUTPUT:
 *
 * $ java FastCollatz 1 10 100 27 971 2000 178913277
 *
 * Longest sequence up to 1 found at 0 with length 0.
 * Longest sequence up to 10 found at 9 with length 19.
 * Longest sequence up to 100 found at 97 with length 118.
 * Longest sequence up to 27 found at 27 with length 111.
 * Longest sequence up to 971 found at 871 with length 178.
 * Longest sequence up to 2,000 found at 1,161 with length 181.
 * Longest sequence up to 178,913,277 found at 169,941,673 with length 953.
 */


public class FastCollatz
{
	static long intRange;
	static int maxInteger;
	static long[] allLenArray;

	public FastCollatz(){};

	public static long getNext( long x )
	{
		if (x == 1) { return 1; }
		if (x % 2 == 0) { return x/2; }
		else { return 3 * x + 1; }
	}

	public static void makeMaxInteger( int i )
	{
		maxInteger = i;
	}

	// Finds lengths of Collatz Sequence for numbers up to maxint.
	// Adds lengths to respective indices in allLenArray so that one does not have to recalculate.
	// (Memoization)
	public void longestLength( int maxint )
	{
		allLenArray = new long[ (int) maxInteger+1 ];

		// Default, fills all indexes to zero.
		long[] lengtharray = new long[2];
		long templength = 0;
		for ( long i = 2; i <= maxInteger; i += 1)
		{
			templength = sequenceLength(i);
			if ( templength > lengtharray[1] )
			{
				lengtharray[0] = i;
				lengtharray[1] = templength;
			}
		}
		DecimalFormat formatter = new DecimalFormat("#,###");
		System.out.println( "Longest sequence up to " + formatter.format(maxInteger) + " found at " + formatter.format((int)lengtharray[0]) + " with length " + Integer.toString((int)lengtharray[1]) + ".");
	}

	public long sequenceLength( long x )
	{
		if ( allLenArray[ (int) x ] > 0) { return allLenArray[ (int) x ]; }
		if (x == 1) { return 1; }

		long origx = x;
		while ( x != 1 )
		{
			allLenArray[ (int) origx ] += 1;
			x=getNext(x);

			if ( x <= maxInteger )
			{
				if ( allLenArray[ (int) x ] > 0)
				{
					allLenArray[ (int) origx ] =  allLenArray[ (int) origx ] + allLenArray[ (int) x ];
					x = 1;
					return allLenArray[ (int) origx ];
				}
			}		
		}
		return allLenArray[ (int) origx ];
	}

	public static void main(String[] args) 
	{
		FastCollatz mc = new FastCollatz();
		for (int i = 0; i < args.length; i+=1)
		{
			String input = args[i];
			int max = Integer.parseInt(input);
			mc.makeMaxInteger( max );
			mc.longestLength( max );
		}
	}
}
