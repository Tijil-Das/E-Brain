import java.util.Scanner;

public class Driver
{
    static Scanner sc=new Scanner(System.in); 
    static String text=Data.chunk1;
    static char[] char_present_in_text;
    static double[][] matrix_scores;
    static int array_length;
    static double learning_rate=0.1;
    static int epochs=30;

    public static void main(String[] args)
    {
        main();
    }

    public static void main()
    {
        Library.fill_the_characters_present_in_text(text);
        char_present_in_text=Library.ch_present;
        array_length=char_present_in_text.length;

        System.out.println("=== Stage 2: Learned Table Model ===");
        System.out.println("Vocabulary size V = " + array_length);
        System.out.println("Theoretical initial loss ln(V) = ln(" + array_length + ") = " + String.format("%.4f", Math.log(array_length)));

        //Step 1: Make V x V table of scores filled with small random numbers
        matrix_scores=Library.make_scores_matrix_and_fill_small_random_values(char_present_in_text);

        //Check initial loss before training
        double starting_loss=Library.compute_loss_on_text(matrix_scores, text, char_present_in_text);
        System.out.printf("Starting loss before training: %.4f%n%n", starting_loss);

        //Step 5: Train for multiple epochs
        System.out.println("--- Starting Training (" + epochs + " epochs, learning rate = " + learning_rate + ") ---");
        double final_loss=starting_loss;
        for(int epoch=1; epoch<=epochs; epoch++)
        {
            double epoch_loss=Library.train_one_epoch(matrix_scores, text, char_present_in_text, learning_rate);
            final_loss=epoch_loss;
            System.out.printf("Epoch %2d / %2d - Average Loss: %.4f%n", epoch, epochs, epoch_loss);
        }

        System.out.println();
        System.out.printf("Training complete! Starting loss: %.4f | Final loss: %.4f%n%n", starting_loss, final_loss);

        //Step 6: Check row for 'q'
        Library.print_row_probabilities('q', matrix_scores);

        //Generate text
        System.out.println("Enter how many characters to generate:");
        long n=300;
        if(sc.hasNextLong())
        {
            n=sc.nextLong();
        }
        else if(sc.hasNextInt())
        {
            n=sc.nextInt();
        }

        System.out.println("\n--- Generating " + n + " characters ---");
        int random_start_index=(int)(Math.random()*array_length);
        char ch=Library.find_ch_at_a_number(random_start_index, char_present_in_text);
        char generated='\0';
        for(long i=1; i<=n; i++)
        {
            if(i==1)
                generated=Library.generate(ch, matrix_scores);
            else
                generated=Library.generate(generated, matrix_scores);

            System.out.print(generated);
        }
        System.out.println("\n");
    }
}
