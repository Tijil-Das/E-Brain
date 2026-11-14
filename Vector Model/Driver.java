import java.util.Scanner;

public class Driver
{
    static Scanner sc = new Scanner(System.in); 
    static String text = Data.chunk1;
    static char[] char_present_in_text;
    static int v;
    static int d = 10;
    static int input_dim = 2 * d;
    static double[][] C;
    static double[][] W;
    static double[] b;
    static double learning_rate = 0.01;
    static int epochs = 20;

    public static void main(String[] args)
    {
        main();
    }

    public static void main()
    {
        Library.fill_the_characters_present_in_text(text);
        char_present_in_text = Library.ch_present;
        v = char_present_in_text.length;

        System.out.println("=== Stage 3: Two-Character Memory Vector Model ===");
        System.out.println("Vocabulary size V = " + v + ", Embedding dimension D = " + d + ", Input dimension = " + input_dim);
        System.out.printf("Theoretical initial loss ln(V) = ln(%d) = %.4f%n%n", v, Math.log(v));

        //Step 1 & 3: Initialize parameters
        C = Library.make_embedding_table(v, d);
        W = Library.make_weight_matrix(input_dim, v);
        b = Library.make_bias_vector(v);

        //Debug test: gradient check on first training example
        int init_c1 = Library.find_ch_number(text.charAt(0), char_present_in_text);
        int init_c2 = Library.find_ch_number(text.charAt(1), char_present_in_text);
        int init_c3 = Library.find_ch_number(text.charAt(2), char_present_in_text);
        Library.debug_gradient_test(C, W, b, init_c1, init_c2, init_c3, d);

        //Starting loss before training
        double starting_loss = Library.compute_loss_on_text(C, W, b, text, char_present_in_text, d);
        System.out.printf("Starting loss before training: %.4f%n%n", starting_loss);

        //Step 5: Train for epochs
        System.out.println("--- Starting Training (" + epochs + " epochs, learning rate = " + learning_rate + ") ---");
        double final_loss = starting_loss;
        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            double epoch_loss = Library.train_one_epoch(C, W, b, text, char_present_in_text, learning_rate, d);
            final_loss = epoch_loss;
            System.out.printf("Epoch %2d / %2d - Average Loss: %.4f%n", epoch, epochs, epoch_loss);
        }

        System.out.println();
        System.out.printf("Training complete! Starting loss: %.4f | Final loss: %.4f%n%n", starting_loss, final_loss);

        //Text generation
        System.out.println("Enter how many characters to generate:");
        long n = 300;
        if(sc.hasNextLong())
        {
            n = sc.nextLong();
        }
        else if(sc.hasNextInt())
        {
            n = sc.nextInt();
        }

        System.out.println("\n--- Generating " + n + " characters ---");
        int r1 = (int)(Math.random() * v);
        int r2 = (int)(Math.random() * v);
        char c1 = Library.find_ch_at_a_number(r1, char_present_in_text);
        char c2 = Library.find_ch_at_a_number(r2, char_present_in_text);
        System.out.print("" + c1 + c2);

        for(long i = 1; i <= n - 2; i++)
        {
            int c1_idx = Library.find_ch_number(c1, char_present_in_text);
            int c2_idx = Library.find_ch_number(c2, char_present_in_text);
            char next_char = Library.sample_next(c1_idx, c2_idx, C, W, b, d);
            System.out.print(next_char);
            c1 = c2;
            c2 = next_char;
        }
        System.out.println("\n");
    }
}
