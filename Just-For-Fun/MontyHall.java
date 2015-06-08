// Author: Alex Walczak 2015

// For choosing random integers:
import java.util.Random;

// For printing arrays:
import java.util.Arrays;
import org.junit.Assert;
import java.util.*;


/* Simulates running the Monty Hall problem a few million times.
 * 
 * STORY:
 * 
 * Behind one of three doors, there is a new car.
 * Behind the other two doors, there are goats.
 * You pick a door first. And then a door with a goat
 * behind it is revealed to you.
 * 
 * You are allowed to switch your choice now.
 * Assuming you want the car, should you switch? 
 * Run this simulation to find out.
 */

public class MontyHall
{

	private int wins = 0;
	private int gamesPlayed = 0;	

	/* Create empty framework (bool array) for doors. */
	private static boolean[] buildDoors()
	{
		boolean[] framework = new boolean[]{false, false, false};
		return framework;
	}

	/* Returns random integer, range inclusive. */
	private static int randInt(int min, int max) 
	{
		Random rand = new Random();
		int randomNum = rand.nextInt((max - min) + 1) + min;
		return randomNum;
	}


	/* Returns index of door with car behind it. */
	private int chooseCarDoor()
	{
		return randInt(0,2);
	}

	/* Puts car at index chooseCarDoor, goats at other doors. */
	private boolean[] chooseGoatDoors(int cd)
	{
		// I let goats = false, car = true.
		boolean[] doors = new boolean[3];
		doors = buildDoors();
		doors[cd] = true;
		return doors;
	}

	/* Guess a door, any door (of the 3)! */
	private int guess()
	{
		return randInt(0,2);
	}

	/* Pick the door to reveal. There is only one possiblility: not guessed and not car.
	 * Avoid the door which has been guessed.
	 */
	private int doorToReveal(int guess, int carDoor)
	{
		for (int doorIndex = 0; doorIndex < 3; doorIndex += 1)
		{
			if (doorIndex != guess && doorIndex != carDoor)
			{
				return doorIndex;
			}
		}
		return 9999999;
	}

	/* Keep track of wins. */
	private void wins(boolean c)
	{
		gamesPlayed+=1;

		if (c==true)
		{
			wins+=1;
		}
	}

	public void playGame(boolean switchDoor)
	{

		/* We'll be removing from this set to make sure our tests work
		 * and switchDoor is done right. 
		 */		
		Set<Integer> possibledoors = new HashSet<Integer>();
		possibledoors.add(0);
		possibledoors.add(1);
		possibledoors.add(2);

		/*  Randomly selects a door indexed 0 to 2. */		
		int carDoor = chooseCarDoor();
		
		/*  Put the car in the correct door, and the goats in the others. */		
		boolean[] doors = chooseGoatDoors(carDoor);

		/*  Randomly selects a door indexed 0 to 2. */		
		int guess = guess();

		/*  If switching, can't choose first guess. */		
		possibledoors.remove(guess);

		/*  This door will be revealed. */
		int revealedDoor = doorToReveal(guess, carDoor);

		/*  If switching, can't choose revealed door. */		
		possibledoors.remove(revealedDoor);


		if (switchDoor == true)
		{
			/*  Makes sure we got rid of revealed door and first guess before making switch. */		
			Assert.assertEquals(possibledoors.size(), 1);

			/*  Change our guess to the door that wasn't revealed. */		
			Iterator it = possibledoors.iterator();
			guess = (int) it.next();
			
		}

		/* Record wins when we guess correctly. */		
		if (carDoor == guess)
		{
			wins(true);
		}

		if (carDoor != guess)
		{
			wins(false);

		}
	}

	/* Runs the simulation "total" times, shows win ratio. */		
	public void playGameNTimes(int total, boolean c)
	{
		for (int count = 0; count < total; count += 1)
		{
			playGame(c);
		}
		getWinRatio();
	}

	/* Shows which strategy (switch or not) is better. */
	public void getWinRatio()
	{
		System.out.println("Total wins: "+Integer.toString(wins)+". Games played: "+Integer.toString(gamesPlayed));
		if (gamesPlayed > 0)
		{	
			float winRatio = (float) wins/gamesPlayed;
			System.out.println("Probability of winning: "+Float.toString(winRatio)+".");
		}

	}

	public static void main(String[] args) 
	{
		System.out.println("Door never switched... ");
		MontyHall game = new MontyHall();
		game.playGameNTimes(4000000, false);

		System.out.print(System.lineSeparator());

		System.out.println("Door swapped for the one not guessed and not revealed... ");
		MontyHall game2 = new MontyHall();
		game2.playGameNTimes(4000000, true);	
	}
}
