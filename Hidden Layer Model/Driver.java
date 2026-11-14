import java.util.Scanner;

public class Driver
{
    static Scanner sc = new Scanner(System.in); 
    static String text = Data.chunk1;
    static char[] char_present_in_text;
    static int v;
    static int n = 4; // Context length (4 characters)
    static int d = 10; // Embedding dimension
    static int h = 100; // Hidden layer size
    static int input_dim = n * d; // 40 numbers
    static double[][] C;
    static double[][] W1;
    static double[] b1;
    static double[][] W2;
    static double[] b2;
    static double learning_rate = 0.01;
    static int epochs = 10;

    public static void main(String[] args)
    {
        main();
    }

    public static void main()
    {
        Library.fill_the_characters_present_in_text(text);
        char_present_in_text = Library.ch_present;
        v = char_present_in_text.length;

        System.out.println("=== Stage 4 - Change B: Hidden Layer (N = 4, H = 100, Tanh) ===");
        System.out.println("Vocabulary size V = " + v + ", Context N = " + n + ", Embedding D = " + d + ", Hidden H = " + h);
        System.out.printf("Theoretical initial loss ln(V) = ln(%d) = %.4f%n%n", v, Math.log(v));

        //Initialize parameters
        C = Library.make_embedding_table(v, d);
        W1 = Library.make_W1_matrix(input_dim, h);
        b1 = Library.make_b1_vector(h);
        W2 = Library.make_W2_matrix(h, v);
        b2 = Library.make_b2_vector(v);

        //Safety Net: Gradient check on W1[0][0]
        int[] init_context = new int[n];
        for(int k = 0; k < n; k++)
            init_context[k] = Library.find_ch_number(text.charAt(k), char_present_in_text);
        int init_target = Library.find_ch_number(text.charAt(n), char_present_in_text);

        double[] grad_check = Library.debug_gradient_test_W1(init_context, init_target, C, W1, b1, W2, b2, d);
        System.out.printf("Gradient Check (W1[0][0]) -> Analytic: %.8f | Numerical: %.8f (Match: %b)%n%n",
            grad_check[0], grad_check[1], Math.abs(grad_check[0] - grad_check[1]) < 1e-5);

        //Starting loss before training
        double starting_loss = Library.compute_loss_on_text(C, W1, b1, W2, b2, text, char_present_in_text, n, d);
        System.out.printf("Starting loss before training: %.4f%n%n", starting_loss);

        //Train
        System.out.println("--- Starting Training (" + epochs + " epochs, learning rate = " + learning_rate + ") ---");
        double final_loss = starting_loss;
        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            double epoch_loss = Library.train_one_epoch(C, W1, b1, W2, b2, text, char_present_in_text, learning_rate, n, d);
            final_loss = epoch_loss;
            System.out.printf("Epoch %2d / %2d - Average Loss: %.4f%n", epoch, epochs, epoch_loss);
        }

        System.out.println();
        System.out.printf("Change B Complete! Starting loss: %.4f | Final loss: %.4f%n%n", starting_loss, final_loss);

        //Text Generation
        System.out.println("Enter how many characters to generate:");
        long num_chars = 300;
        if(sc.hasNextLong())
        {
            num_chars = sc.nextLong();
        }
        else if(sc.hasNextInt())
        {
            num_chars = sc.nextInt();
        }

        System.out.println("\n--- Generating " + num_chars + " characters ---");
        int[] current_context = new int[n];
        for(int k = 0; k < n; k++)
        {
            current_context[k] = (int)(Math.random() * v);
            System.out.print(char_present_in_text[current_context[k]]);
        }

        for(long i = 1; i <= num_chars - n; i++)
        {
            char next_char = Library.sample_next(current_context, C, W1, b1, W2, b2, d);
            System.out.print(next_char);

            //Shift context left by 1
            for(int k = 0; k < n - 1; k++)
            {
                current_context[k] = current_context[k + 1];
            }
            current_context[n - 1] = Library.find_ch_number(next_char, char_present_in_text);
        }
        System.out.println("\n");
    }
}
