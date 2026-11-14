import java.util.concurrent.ThreadLocalRandom;

public class Library 
{
    //We need to declare an array to hold the characters that are present in the text
    static char[] ch_present;

    public static void fill_the_characters_present_in_text(String data)
    {
        //First find all unique characters in the text
        char[] temp=new char[256];
        int c=0;
        String text=data;
        for(int i=0; i<text.length(); i++)//Iterate through each letter and fill the temp array if not seen before
        {
            char ch=text.charAt(i);
            if(check_if_char_already_filled(ch, temp, c)==false)
            {
                temp[c]=ch;
                c++;
            }
        }

        //Now initialize ch_present with the exact count of unique characters found
        ch_present=new char[c];
        for(int i=0; i<c; i++)
        {
            ch_present[i]=temp[i];
        }
    }

    public static int find_how_many_array_cells_filled()
    {
        if(ch_present==null)
            return 0;
        int c=0;
        for(int i=0; i<ch_present.length; i++)
        {
            if(ch_present[i] != '\0')
                c++;
        }
        return c;
    }

    public static boolean check_if_char_already_filled(char ch, char[] array, int length)
    {
        boolean flag=false;
        for(int i=0; i<length; i++)
        {
            if(array[i]==ch)
            {
                flag=true;
                break;
            }
        }
        return flag;
    }

    public static int find_ch_number(char ch, char[] array)//find the zero-based position of a character in an array. if character not present, return -1 
    {
        for(int i=0; i<array.length; i++)
        {
            if(array[i]==ch)
                return i;
        }
        return -1;
    }

    public static Character find_ch_at_a_number(int n, char[] array)//inputs the array and the zero-based position number and outputs the character in that array at that position
    {
        if(n<0 || n>=array.length)
            return null;
        return array[n];
    }

    //Step 1: Make a V x V table of scores, filled with small random numbers (-0.1 to 0.1)
    public static double[][] make_scores_matrix_and_fill_small_random_values(char[] array)
    {
        int count=array.length;
        double[][] scores=new double[count][count];

        for(int i=0; i<count; i++)
        {
            for(int j=0; j<count; j++)
            {
                //Pick a small random double between -0.1 and 0.1
                scores[i][j]=ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
            }
        }
        return scores;
    }

    //Step 2: Turn one row into probabilities (softmax)
    public static double[] calculate_softmax(double[] row_scores)
    {
        int v=row_scores.length;

        //Subtract the biggest score from every score (this prevents overflow)
        double max_score=row_scores[0];
        for(int i=1; i<v; i++)
        {
            if(row_scores[i]>max_score)
                max_score=row_scores[i];
        }

        //Raise e to the power of each score
        double[] exp_scores=new double[v];
        double total_exp_sum=0.0;
        for(int i=0; i<v; i++)
        {
            exp_scores[i]=Math.exp(row_scores[i]-max_score);
            total_exp_sum=total_exp_sum+exp_scores[i];
        }

        //Divide each by the total of all of them
        double[] probabilities=new double[v];
        for(int i=0; i<v; i++)
        {
            probabilities[i]=exp_scores[i]/total_exp_sum;
        }

        return probabilities;
    }

    //Step 3: Measure how wrong it was: Loss = -ln(p)
    public static double calculate_loss(double p)
    {
        return -Math.log(p);
    }

    //Step 4: Nudge the scores
    public static void nudge_scores(double[] row_scores, double[] probabilities, int correct_column, double learning_rate)
    {
        for(int j=0; j<row_scores.length; j++)
        {
            //gradient is: (its probability) minus (1 if it's the correct character, else 0)
            double target=0.0;
            if(j==correct_column)
                target=1.0;

            double gradient=probabilities[j]-target;

            //score = score - learningRate * gradient
            row_scores[j]=row_scores[j]-(learning_rate*gradient);
        }
    }

    //Step 5: Train for one epoch over the entire text
    public static double train_one_epoch(double[][] scores, String text, char[] array, double learning_rate)
    {
        double total_loss=0.0;
        int total_pairs=text.length()-1;

        for(int i=0; i<total_pairs; i++)
        {
            char current_char=text.charAt(i);
            char next_char=text.charAt(i+1);

            int current_row=find_ch_number(current_char, array);
            int next_column=find_ch_number(next_char, array);

            //Step 2: Turn current row into probabilities (softmax)
            double[] probabilities=calculate_softmax(scores[current_row]);

            //Step 3: Measure how wrong it was
            double p=probabilities[next_column];
            double loss=calculate_loss(p);
            total_loss=total_loss+loss;

            //Step 4: Nudge the scores for that row
            nudge_scores(scores[current_row], probabilities, next_column, learning_rate);
        }

        double average_loss=total_loss/total_pairs;
        return average_loss;
    }

    //Compute average loss on the text without nudging (used to check starting loss before any training)
    public static double compute_loss_on_text(double[][] scores, String text, char[] array)
    {
        double total_loss=0.0;
        int total_pairs=text.length()-1;

        for(int i=0; i<total_pairs; i++)
        {
            char current_char=text.charAt(i);
            char next_char=text.charAt(i+1);

            int current_row=find_ch_number(current_char, array);
            int next_column=find_ch_number(next_char, array);

            double[] probabilities=calculate_softmax(scores[current_row]);
            double p=probabilities[next_column];
            double loss=calculate_loss(p);
            total_loss=total_loss+loss;
        }

        return total_loss/total_pairs;
    }

    //Sample next character from the row probabilities
    public static char generate(char starter_char, double[][] scores)
    {
        int row_of_starter_char=find_ch_number(starter_char, ch_present);
        if(row_of_starter_char==-1)
            row_of_starter_char=0;

        double[] probabilities=calculate_softmax(scores[row_of_starter_char]);
        double random=ThreadLocalRandom.current().nextDouble();

        int i=0;
        double sum=0.0;
        for(i=0; i<ch_present.length; i++)
        {
            sum=sum+probabilities[i];
            if(sum>random)
                break;
        }

        if(i>=ch_present.length)
            i=ch_present.length-1;

        char generated=find_ch_at_a_number(i, ch_present);
        return generated;
    }

    //Step 6 check: Print top probabilities for a character (e.g. 'q')
    public static void print_row_probabilities(char ch, double[][] scores)
    {
        int row=find_ch_number(ch, ch_present);
        if(row==-1)
        {
            System.out.println("Character '" + ch + "' not found in vocabulary.");
            return;
        }

        double[] probabilities=calculate_softmax(scores[row]);
        System.out.println("--- Probabilities for row '" + ch + "' ---");
        
        //Find the character with the maximum probability
        int max_index=0;
        double max_p=probabilities[0];
        for(int i=1; i<ch_present.length; i++)
        {
            if(probabilities[i]>max_p)
            {
                max_p=probabilities[i];
                max_index=i;
            }
        }
        System.out.printf("Highest probability next character after '%c' is '%c' with probability: %.4f (%.2f%%)%n", 
            ch, ch_present[max_index], max_p, max_p*100.0);

        //Also specifically check 'u'
        int u_index=find_ch_number('u', ch_present);
        if(u_index != -1)
        {
            System.out.printf("Probability of 'u' after '%c': %.4f (%.2f%%)%n", 
                ch, probabilities[u_index], probabilities[u_index]*100.0);
        }

        System.out.println();
    }
}
